import pre_commit
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.ir import Operation, SSAValue, Block, Attribute
from xdsl.context import Context
from xdsl.passes import ModulePass
import spensor.dialects.spensor_dialect as spensor
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)

from spensor.dialects.spensor_dialect import (
    getConstantFromSSA,
    toTupleInt,
)
from spensor.utils.spensor_global_analysis import Memory, ConstantLoop
import spensor.utils.spensor_global_analysis as spensor_global

op_to_parent_loop: dict[Operation, ConstantLoop] = {}
op_to_offset_loop_vars: dict[
    Operation, tuple[tuple[int, ConstantLoop] | None, ...]
] = {}
loop_var_to_op_offsets: dict[tuple[int, ConstantLoop], list[tuple[int, Operation]]] = {}
increase_bound_record: dict[ConstantLoop, tuple[tuple[ConstantLoop, int], ...]] = {}
loop_without_subviews: set[ConstantLoop] = set()


def insertOpToLoop(
    op: Operation,
    loop: ConstantLoop,
    offsets_to_loop_vars: tuple[tuple[int, ConstantLoop] | None, ...],
):
    """
    Attachs an operation into a constant loop.
    User must provide usages of how the operaiton uses loop induction variables

    Example:
    - split(<4x4xf32>, dim = 1, num_partition = 2) -> <4x2xf32>

    It expends to a constant loop with upper bounds [2]

    Expected offset_to_loop_vars: [None, [0, loop]]
    Because the first dimension doesn't use any variable but the second does
    """
    global op_to_parent_loop
    global op_to_offset_loop_vars
    global loop_var_to_op_offsets
    loop.ops.append(op)
    op_to_offset_loop_vars[op] = offsets_to_loop_vars
    op_to_parent_loop[op] = loop
    for i, loop_var in enumerate(offsets_to_loop_vars):
        if loop_var is not None:
            if loop_var not in loop_var_to_op_offsets:
                loop_var_to_op_offsets[loop_var] = []
            loop_var_to_op_offsets[loop_var].append((i, op))


def increaseBound(loop: ConstantLoop, parent_loop: ConstantLoop, nth: int):
    """
    Increases the nth upper bound of a child loop and decreases the upper bound
    of the parent loop at the same index. 
    """
    parent_bounds = list(parent_loop.upper_bounds)
    bounds = list(loop.upper_bounds)
    global op_to_offset_loop_vars
    global loop_var_to_op_offsets
    if parent_bounds[nth] != 1:
        bounds[nth] = parent_bounds[nth]
        parent_bounds[nth] = 1
        parent_loop.upper_bounds = tuple(parent_bounds)
        loop.upper_bounds = tuple(bounds)


def getNestedLoop(loop: ConstantLoop) -> list[ConstantLoop]:
    """
    Collects a list of ConstantLoop instances from the given loop to its ancestors.
    """
    result = [loop]
    cur_loop = loop.parent_loop
    while cur_loop is not None:
        result = [cur_loop] + result
        cur_loop = cur_loop.parent_loop
    return result


def getParentLoopBySSAValue(val: SSAValue) -> ConstantLoop | None:
    owner = val.owner
    if isinstance(owner, Block):
        return None
    return op_to_parent_loop.get(owner, None)


def checkLoopBoundEq(const1: ConstantLoop, const2: ConstantLoop) -> bool:
    """
    Check if two constant loop have same upper bounds. 
    Ignore dimensions with upper bound 1
    """
    loop_bounds1 = [x for x in const1.upper_bounds if x != 1]
    loop_bounds2 = [x for x in const2.upper_bounds if x != 1]
    return loop_bounds1 == loop_bounds2 and (
        const1.memory_tag is None
        or const2.memory_tag is None
        or const1.memory_tag == const2.memory_tag
    )


def getLoopIndexMapping(const1: ConstantLoop, const2: ConstantLoop) -> dict[int, int]:
    """
    Get a indice mapping from loop1 to loop2, ignoring dimensions with upper bound 1
    """
    loop_bounds1 = [(i, x) for i, x in enumerate(const1.upper_bounds) if x != 1]
    loop_bounds2 = [(i, x) for i, x in enumerate(const2.upper_bounds) if x != 1]
    assert len(loop_bounds1) == len(loop_bounds2)
    result: dict[int, int] = {}
    for loop_idx2, loop_idx1 in zip(loop_bounds2, loop_bounds1):
        result[loop_idx2[0]] = loop_idx1[0]
    return result


def fuseLoop(loop1: ConstantLoop, loop2: ConstantLoop) -> ConstantLoop:
    """
    Fuses loop2 into loop1, combining their parent loops as well.
    This fusion includes:
    1. Update loop induction variable usages
    2. Append all operations in loop2 into loop1
    3. Update operation usages of loop vars.
    """
    nested_loop1, nested_loop2 = getNestedLoop(loop1), getNestedLoop(loop2)
    if len(nested_loop1) < len(nested_loop2):
        nested_loop1, nested_loop2 = nested_loop2, nested_loop1
    # Assume  len(nested_loop1) >= len(nested_loop2)
    length = len(nested_loop2)
    global loop_to_ops
    global op_to_parent_loop
    global op_to_offset_loop_vars
    global loop_var_to_op_offsets
    for i in range(length):
        loop1 = nested_loop1[i]
        loop2 = nested_loop2[i]
        assert checkLoopBoundEq(loop1, loop2)
        loop_index_mapping = getLoopIndexMapping(loop1, loop2)
        if loop1.memory_tag is None:
            loop1.memory_tag = loop2.memory_tag
        while len(loop2.ops) >= 1:
            op = loop2.ops.pop()
            op_to_parent_loop[op] = loop1
            loop1.ops.append(op)

            use_loop_vars = list(op_to_offset_loop_vars[op])
            for nth_dim, loop_var in enumerate(use_loop_vars):
                if loop_var is not None and loop_var[1] == loop2:
                    use_loop_vars[nth_dim] = (loop_index_mapping[loop_var[0]], loop1)
            op_to_offset_loop_vars[op] = tuple(use_loop_vars)

        for i in range(len(loop2.upper_bounds)):
            use_loop2 = (i, loop2)
            if use_loop2 in loop_var_to_op_offsets:
                use_loop1 = (loop_index_mapping[i], loop1)
                if use_loop1 not in loop_var_to_op_offsets:
                    loop_var_to_op_offsets[use_loop1] = []
                loop_var_to_op_offsets[use_loop1] += loop_var_to_op_offsets[use_loop2]
                loop_var_to_op_offsets[use_loop2] = []
    return nested_loop1[len(nested_loop2) - 1]


def getMemoryTag(typ: Attribute):
    if isinstance(typ, spensor.NDSpensorType):
        return spensor_global.memory_mapping[typ.get_memory().memory_name]
    elif isinstance(typ, spensor.SpensorType):
        return spensor_global.memory_mapping[typ.memory.memory_name]
    else:
        assert False and "Cannot get memory info"


def getTensorShape(typ: Attribute):
    if isinstance(typ, spensor.NDSpensorType):
        return typ.spensor.element_type.element_type.get_shape()
    elif isinstance(typ, spensor.SpensorType):
        return typ.element_type.get_shape()
    else:
        assert False and "Cannot get shape info"


class SpensorSplitPattern(RewritePattern):
    """
    Constructs a constant loop and attachs it with SplitOp

    Example:
    - split(<4x4xf32>, dim=1, num_parttion = 2)
    Result:
    - ConstantLoop(upper_bounds = [2])
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.SplitOp, rewriter: PatternRewriter):
        source = op.source
        if op.result not in op_to_parent_loop:
            if source in op_to_parent_loop:
                assert False and "Current Split op only works on SpensorType"
            else:
                num_partitions = getConstantFromSSA(op.num_partitions)
                dim = getConstantFromSSA(op.dim)
                result_shape = getTensorShape(op.result.type)
                loop = ConstantLoop((num_partitions,))
                offset_loop_vars: list[tuple[int, ConstantLoop] | None] = [
                    None for _ in result_shape
                ]
                offset_loop_vars[dim] = (0, loop)
                insertOpToLoop(op, loop, tuple(offset_loop_vars))
            rewriter.has_done_action = True


class SpensorSplitAllPattern(RewritePattern):
    """
    Constructs a constant loop and attachs it with SplitOp

    Example:
    - SplitAll(<4x4xf32>, num_parttions = [2,4])
    Result:
    - ConstantLoop(upper_bounds = [2,4])
    """    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.SplitAllOp, rewriter: PatternRewriter):
        source = op.source.owner
        if source in op_to_parent_loop:
            assert False and "Current Split op only works on SpensorType"
        else:
            num_partitions = toTupleInt(op.num_partitions)
            result_partitions = [dim for dim in num_partitions if dim != 1]
            if len(result_partitions) == 0:
                result_partitions = [1]
            else:
                result_partitions = [dim for dim in num_partitions]  # if dim != 1]
            loop = ConstantLoop(tuple(result_partitions))
            offset_loop_vars: list[tuple[int, ConstantLoop] | None] = []
            loop_var_idx = 0
            for partition in num_partitions:
                if partition != 1:
                    offset_loop_vars.append((loop_var_idx, loop))
                    loop_var_idx += 1
                else:
                    offset_loop_vars.append(None)
                    loop_var_idx += 1
            insertOpToLoop(op, loop, tuple(offset_loop_vars))


class SpensorMovePattern(RewritePattern):
    """
    Checks if a move operation is valid and attachs it to certain constant loop.
    If the memory tag of destination is different from the loop,
    it creates a new constant loop with upper bounds [1]

    Example: 
    - move(buff, L1)

    Result:
    - buff is at L2 
      move is attached with Constant Loop([1], "L1")
    
    Example:
    - move(buff, L2)

    Result:
    - buff is at L2 
      move is attached with buff's constant loop
    """       
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MoveOp, rewriter: PatternRewriter):
        source = op.source.owner
        assert source in op_to_parent_loop
        to_memory_tag = getMemoryTag(op.result.type)
        from_memory_tag = getMemoryTag(op.source.type)

        # If parent loop has no memory tag, which means it has no any memory related operation
        # Attach to_memory_tag on it
        assert isinstance(source, Operation)
        if op_to_parent_loop[source].memory_tag is None:
            op_to_parent_loop[source].memory_tag = to_memory_tag
        parent_loop = op_to_parent_loop[source]
        parent_loop_tag = parent_loop.memory_tag
        assert parent_loop_tag is not None

        if to_memory_tag == parent_loop_tag:
            # Current at L2
            # Move L3, L2
            # Move L1, L2 (Invalid) since current at L2
            if from_memory_tag.name in parent_loop_tag.allowing_load:
                insertOpToLoop(op, op_to_parent_loop[source], ())
            else:
                assert False and "move op doesn't support"
        elif from_memory_tag == parent_loop_tag:
            # Current at L2
            if to_memory_tag.name in parent_loop_tag.allowing_store:
                # Move L2, L3
                # no extra op needed
                insertOpToLoop(op, op_to_parent_loop[source], ())
            else:
                # Move L2, L1
                # This case should be handled by L1
                # since L2 can't store to L1 and only L1 can load L2
                assert parent_loop_tag.name in to_memory_tag.allowing_load
                # Make a new neasted loop with tag but still use parent's ind var
                parent_bounds = parent_loop.upper_bounds
                new_bounds = tuple([1 for _ in parent_bounds])
                loop = ConstantLoop(
                    new_bounds, to_memory_tag, op_to_parent_loop[source]
                )
                insertOpToLoop(op, loop, ())
                global loop_without_subviews
                loop_without_subviews.add(loop)
        else:
            # Current at L1
            # move L2, L3
            # Find on L1's parent loop with memory_tag == L2 or L3, insert it at that level
            parent_loop = parent_loop.parent_loop
            parent_loop_tag = parent_loop_tag
            while parent_loop is not None and (
                from_memory_tag == parent_loop_tag or to_memory_tag == parent_loop_tag
            ):
                insertOpToLoop(op, parent_loop, ())
            assert parent_loop is not None


def moveOpBeforeOp(
    op: Operation, before: Operation, prev_before: Operation, loop: ConstantLoop
):
    """
    Moves an operation before another operation, and its operand recursively as well

    Examples:
    - block:
        prev_before
        |
        V
        before
        |
        V
        op

    Result:
    - block:
        prev_before
        |
        V
        op
        |
        V
        before    
    """
    if before.is_before_in_block(op):
        op.detach()
        before_parenet_block = before.parent_block()
        assert before_parenet_block is not None
        before_parenet_block.insert_op_after(op, prev_before)
        for operand in op.operands:
            if isinstance(operand.owner, Operation) and (not operand.owner in loop.ops):
                moveOpBeforeOp(operand.owner, before, prev_before, loop)


def moveOpBeforeLoop(op: Operation, loop: ConstantLoop):
    """
    Moves an operation to the front of a loop, and its operand recursively as well 
    due to SSA property.
    """
    before_op = loop.ops[0]
    assert before_op is not None
    prev_before_op = before_op.prev_op
    if prev_before_op is not None:
        moveOpBeforeOp(op, before_op, prev_before_op, loop)


def simpleBinaryOpPattern(lhs: Operation, rhs: Operation, op: Operation):
    """
    For a binary operation, merges lhs and rhs loops first if they are different 
    Attachs the operation to lhs's loop
    """
    lhs_loop, rhs_loop = op_to_parent_loop[lhs], op_to_parent_loop[rhs]
    if lhs_loop != rhs_loop:
        assert checkLoopBoundEq(lhs_loop, rhs_loop)
        fused_loop = fuseLoop(lhs_loop, rhs_loop)
        insertOpToLoop(op, fused_loop, ())
        moveOpBeforeLoop(lhs, fused_loop)
        moveOpBeforeLoop(rhs, fused_loop)
    else:
        insertOpToLoop(op, op_to_parent_loop[lhs], ())


class SpensorAddPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.AddOp, rewriter: PatternRewriter):
        lhs, rhs = op.lhs.owner, op.rhs.owner
        assert isinstance(lhs, Operation)
        assert isinstance(rhs, Operation)
        simpleBinaryOpPattern(lhs, rhs, op)


class SpensorMatmulPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MatmulOp, rewriter: PatternRewriter):
        lhs, rhs = op.lhs.owner, op.rhs.owner
        assert isinstance(lhs, Operation)
        assert isinstance(rhs, Operation)
        simpleBinaryOpPattern(lhs, rhs, op)


class SpensorReduceSumPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.ReduceSumOp, rewriter: PatternRewriter):
        source = op.operand.owner
        assert isinstance(source, Operation)
        insertOpToLoop(op, op_to_parent_loop[source], ())


class SpensorMoveToPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MoveToOp, rewriter: PatternRewriter):
        source, dest = op.source.owner, op.destination.owner
        assert isinstance(source, Operation)
        assert isinstance(dest, Operation)
        source_loop, dest_loop = op_to_parent_loop[source], op_to_parent_loop[dest]
        if source_loop != dest_loop:
            fused_loop = fuseLoop(source_loop, dest_loop)
            insertOpToLoop(op, fused_loop, ())
        else:
            insertOpToLoop(op, source_loop, ())


def getLoopVarUses(loop: ConstantLoop, parent_loop: ConstantLoop):
    """
    Extracts loop vars from a child and parent loop.
    One non-one upper bound must be either the child or parent loop.
    """
    bounds = loop.upper_bounds
    parent_bounds = parent_loop.upper_bounds
    use_loop_vars: list[tuple[int, ConstantLoop]] = []
    for i in range(len(bounds)):
        assert (bounds[i] == 1) ^ (parent_bounds[i] == 1)
        if bounds[i] == 1:
            use_loop_vars.append((i, parent_loop))
        else:
            use_loop_vars.append((i, loop))
    return use_loop_vars


class NDSpensorCombinePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.NDCombineOp, rewriter: PatternRewriter):
        nd_dim = getConstantFromSSA(op.nd_dim)
        dim = getConstantFromSSA(op.dim)
        source = op.source.owner
        assert isinstance(source, Operation)
        source_loop = op_to_parent_loop[source]
        parent_loop = source_loop.parent_loop
        assert parent_loop is not None

        increaseBound(source_loop, parent_loop, nd_dim)
        record = (*increase_bound_record.get(parent_loop, ()), (source_loop, nd_dim))
        increase_bound_record[parent_loop] = record
        result_shape = getTensorShape(op.result.type)

        use_loop_vars: list[tuple[int, ConstantLoop] | None] = [
            None for _ in result_shape
        ]
        use_loop_vars[dim] = (nd_dim, source_loop)
        insertOpToLoop(op, parent_loop, tuple(use_loop_vars))


class NDSpensorReducePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.NDReduceOp, rewriter: PatternRewriter):
        source_type = op.source.type
        assert isinstance(source_type, spensor.NDSpensorType)
        reduce_dim = toTupleInt(op.reduce_dim)

        source = op.source.owner
        assert isinstance(source, Operation)
        source_loop = op_to_parent_loop[source]
        parent_loop = source_loop.parent_loop
        assert parent_loop is not None

        for nd_dim in reduce_dim:
            increaseBound(source_loop, parent_loop, nd_dim)
            record = (
                *increase_bound_record.get(parent_loop, ()),
                (source_loop, nd_dim),
            )
            increase_bound_record[parent_loop] = record

        result_shape = getTensorShape(op.result.type)
        use_loop_vars: list[tuple[int, ConstantLoop] | None] = [
            None for _ in result_shape
        ]
        insertOpToLoop(op, parent_loop, tuple(use_loop_vars))


class NDSpensorRepeatPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.NDRepeatOp, rewriter: PatternRewriter):
        repeats = getConstantFromSSA(op.repeats)
        dim = getConstantFromSSA(op.dim)
        source = op.source.owner
        assert isinstance(source, Operation)
        source_loop = op_to_parent_loop[source]
        upper_bounds = source_loop.upper_bounds

        for i in range(dim, len(upper_bounds)):
            use_loop_var = (i, source_loop)
            use_new_loop_var = (i + 1, source_loop)
            for sub_offset, sub_op in loop_var_to_op_offsets.get(use_loop_var, []):
                offset_loop_vars = list(op_to_offset_loop_vars[sub_op])
                offset_loop_vars[sub_offset] = use_new_loop_var
                op_to_offset_loop_vars[sub_op] = tuple(offset_loop_vars)
            loop_var_to_op_offsets[use_new_loop_var] = loop_var_to_op_offsets[
                use_loop_var
            ]
            del loop_var_to_op_offsets[use_loop_var]

        new_upper_bounds = upper_bounds[: dim - 1] + (repeats,) + upper_bounds[dim:]
        source_loop.upper_bounds = new_upper_bounds
        result_shape = getTensorShape(op.result.type)
        use_loop_vars: list[tuple[int, ConstantLoop] | None] = [
            None for _ in result_shape
        ]
        insertOpToLoop(op, source_loop, tuple(use_loop_vars))


class ParallelAnalysis(ModulePass):
    name = "parallel_analysis"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    SpensorSplitPattern(),
                    SpensorMovePattern(),
                    SpensorSplitAllPattern(),
                    SpensorAddPattern(),
                    SpensorMatmulPattern(),
                    SpensorMoveToPattern(),
                    SpensorReduceSumPattern(),
                    NDSpensorCombinePattern(),
                    NDSpensorReducePattern(),
                    NDSpensorRepeatPattern(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        walker.rewrite_module(op)
        spensor_global.op_to_parent_loop = op_to_parent_loop
        spensor_global.op_to_offset_loop_vars = op_to_offset_loop_vars
        spensor_global.loop_var_to_op_offsets = loop_var_to_op_offsets
        spensor_global.increase_bound_record = increase_bound_record
        spensor_global.loop_without_subviews = loop_without_subviews
