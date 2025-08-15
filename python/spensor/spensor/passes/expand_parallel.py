import enum
from functools import reduce
from operator import sub
from xdsl.dialects.builtin import (
    FloatAttr,
    ModuleOp,
    IndexType,
    AnyFloat,
    MemRefType,
    ArrayAttr,
    TensorType,
    f32,
    FunctionType,
    StringAttr,
    AffineMapAttr,
    IntAttr,
    IntegerAttr,
    DenseArrayBase,
)
from xdsl.ir import Operation, SSAValue, Block, Attribute, BlockArgument, Region, Use
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.dialects import func, scf, memref, arith, affine, linalg
from xdsl.ir.affine import AffineMap
from xdsl.irdl.operations import VarOperand
import spensor.dialects.spensor_dialect as spensor
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from math import prod
from xdsl.utils.hints import isa
from xdsl.rewriter import InsertPoint

from spensor.dialects.spensor_dialect import (
    AllocSpensorOp,
    getConstantFromSSA,
    toTupleInt,
)
from spensor.utils.spensor_util.spensor_util import (
    addFront,
    getIndexConstant,
    getMemorySpace,
    getNestedForLoop,
    getShape,
    getConstantOpByIndex,
)
from spensor.utils.spensor_global_analysis import ConstantLoop
import spensor.utils.spensor_global_analysis as spensor_global

from spensor.passes.parallel_analysis import getMemoryTag


constant_loop_to_parallel_op: dict[ConstantLoop, scf.ParallelOp] = {}
# When we update loop bounds, we need to udpate the subview as well
# This dict maintains the operands used in a subview
# parallel_op_to_subview_operands: dict[scf.ParallelOp, ] = {}


zero_op = getConstantOpByIndex(0)
one_op = getConstantOpByIndex(1)
zero = zero_op.result
one = one_op.result


def makeParallelOpFromConstantLoop(loop: ConstantLoop) -> scf.ParallelOp:
    global constant_loop_to_parallel_op
    if loop not in constant_loop_to_parallel_op:
        upper_bounds = [getConstantOpByIndex(bound) for bound in loop.upper_bounds]
        block = Block([], arg_types=[IndexType() for _ in upper_bounds])
        parallel_op = scf.ParallelOp(
            [zero for _ in upper_bounds],
            upper_bounds,
            [one for _ in upper_bounds],
            Region(block),
        )
        constant_loop_to_parallel_op[loop] = parallel_op
        if loop.memory_tag is not None:
            parallel_op.attributes["memory_tag"] = loop.memory_tag.name
    return constant_loop_to_parallel_op[loop]


def getParallelOp(op: Operation, rewriter: PatternRewriter) -> tuple[Block, bool]:
    op_to_parent_loop = spensor_global.op_to_parent_loop
    constant_loop = op_to_parent_loop[op]
    if constant_loop not in constant_loop_to_parallel_op:
        parallel_op = makeParallelOpFromConstantLoop(constant_loop)
        constant_loop_to_parallel_op[constant_loop] = parallel_op
        # Insert parallel op
        if constant_loop.parent_loop is None:
            rewriter.insert_op_before_matched_op(parallel_op)
        else:
            parent_loop_op = constant_loop_to_parallel_op[constant_loop.parent_loop]
            parent_loop_body = parent_loop_op.body.block
            parent_loop_body.add_op(parallel_op)
            rewriter.has_done_action = True
    result_op = constant_loop_to_parallel_op[constant_loop]
    return result_op.body.block, constant_loop in spensor_global.loop_without_subviews


def getDelinearizeIndexAffineMap(dest_shape: tuple[int, ...]) -> tuple[AffineMap, ...]:
    """
    This version assumes the input is already delinearized variable
    """
    lambda_expr = "lambda x: (x*{dim},)"

    result: list[AffineMap] = [
        AffineMap.from_callable(
            eval(lambda_expr.format(dim=dim)), dim_symbol_split=(0, 1)
        )
        for dim in dest_shape
    ]

    return tuple(result)


def getIdentityIndexAffineMap() -> AffineMap:
    return AffineMap.identity(0, symbolic_rank=1)


def replaceShapeInSpensorType(
    spensor_type: spensor.SpensorType, new_shape: tuple[int, ...]
):
    new_tensor_type = TensorType(spensor_type.element_type.element_type, new_shape)
    return spensor.SpensorType(new_tensor_type, spensor_type.memory)


def getSubViewOpWithSSA(
    source: SSAValue,
    val_list: tuple[SSAValue | None, ...],
    source_type: spensor.SpensorType,
    dest_type: spensor.SpensorType,
) -> list[Operation]:
    """
    This version assumes the input is already delinearized variable
    """

    dest_shape = dest_type.element_type.get_shape()
    source_shape = source_type.element_type.get_shape()
    strides = [one for _ in source_shape]
    offsets = [zero for _ in source_shape]
    dest_sizes = [getConstantOpByIndex(dim).result for dim in dest_shape]
    tilesize_ops: list[Operation] = []
    len_diff = len(offsets) - len(dest_sizes)
    dest_sizes = ([one] * len_diff) + dest_sizes

    assert len(val_list) == len(offsets)
    affine_maps = (
        getIdentityIndexAffineMap(),
    ) * len_diff + getDelinearizeIndexAffineMap(dest_shape)
    for nth_dim, loop_var in enumerate(val_list):
        if loop_var is not None:
            assert isinstance(loop_var, BlockArgument)
            dynamic_offset_op = affine.ApplyOp(
                [loop_var], AffineMapAttr(affine_maps[nth_dim])
            )
            tilesize_ops += [dynamic_offset_op]
            offsets[nth_dim] = dynamic_offset_op.result

    subview_op = spensor.SubviewOp(source, offsets, dest_sizes, strides, dest_type)

    return tilesize_ops + [subview_op]


def getSubViewOp(
    source: SSAValue,
    cur_op: Operation,
    source_type: spensor.SpensorType,
    dest_type: spensor.SpensorType,
) -> list[Operation]:
    """
    This version assumes the input is already delinearized variable
    """

    dest_shape = dest_type.element_type.get_shape()

    ssa_list: list[SSAValue | None] = [None for _ in dest_shape]
    op_to_offset_loop_vars = spensor_global.op_to_offset_loop_vars
    for nth_dim, loop_var in enumerate(op_to_offset_loop_vars[cur_op]):
        if loop_var is not None:
            nth_loop_var, constant_loop = loop_var
            parallel_op = constant_loop_to_parallel_op[constant_loop]
            nth_arg = parallel_op.body.block.args[nth_loop_var]
            ssa_list[nth_dim] = nth_arg

    return getSubViewOpWithSSA(source, tuple(ssa_list), source_type, dest_type)


def getNewSubviewShape(
    cur_op: Operation, dest_type: spensor.SpensorType
) -> tuple[int, ...]:
    dest_shape = dest_type.element_type.get_shape()
    dest_shape_list = list(dest_shape)

    op_to_offset_loop_vars = spensor_global.op_to_offset_loop_vars
    for nth_dim, loop_var in enumerate(op_to_offset_loop_vars[cur_op]):
        if loop_var is not None:
            nth_loop_var, constant_loop = loop_var
            bound_records = spensor_global.increase_bound_record.get(constant_loop)
            if bound_records is not None:
                for loop, index in bound_records:
                    if index == nth_loop_var:
                        dest_shape_list[nth_dim] *= loop.upper_bounds[nth_loop_var]
    dest_shape = tuple(dest_shape_list)
    return tuple(dest_shape_list)


class SpensorSplitPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.SplitOp, rewriter: PatternRewriter):
        dest_type = op.result.type.spensor.element_type
        source_type = op.source.type
        assert isinstance(source_type, spensor.SpensorType)
        parallel_block, _ = getParallelOp(op, rewriter)

        dest_shape = getNewSubviewShape(op, dest_type)
        dest_type = replaceShapeInSpensorType(dest_type, dest_shape)

        subview_ops = getSubViewOp(op.source, op, source_type, dest_type)
        parallel_block.add_ops(subview_ops)
        rewriter.replace_all_uses_with(op.result, subview_ops[-1].results[0])

        rewriter.erase_matched_op()


class SpensorSplitAllPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.SplitAllOp, rewriter: PatternRewriter):
        dest_type = op.result.type.spensor.element_type
        source_type = op.source.type
        assert isinstance(source_type, spensor.SpensorType)
        parallel_block, _ = getParallelOp(op, rewriter)
        dest_shape = getNewSubviewShape(op, dest_type)
        dest_type = replaceShapeInSpensorType(dest_type, dest_shape)

        subview_ops = getSubViewOp(op.source, op, source_type, dest_type)
        parallel_block.add_ops(subview_ops)

        rewriter.replace_all_uses_with(op.result, subview_ops[-1].results[0])
        rewriter.erase_matched_op()


def getSSAListFromParentSubview(op: spensor.MoveOp) -> list[SSAValue]:
    # I haven't considered it carefully
    # This function follows the parent and find the subview
    owner = op.source.owner
    if isinstance(owner, spensor.SubviewOp):
        return list(owner.offsets)
    elif isinstance(owner, spensor.MoveOp):
        return getSSAListFromParentSubview(owner)
    else:
        return []


class SpensorMovePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MoveOp, rewriter: PatternRewriter):
        result_type = op.result.type
        source = op.source
        source_type = op.source.type
        assert isinstance(source_type, spensor.SpensorType)
        assert isinstance(result_type, spensor.NDSpensorType)

        parallel_block, need_subviews = getParallelOp(op, rewriter)

        if need_subviews:
            constant_loop = spensor_global.op_to_parent_loop[op]
            assert constant_loop in spensor_global.loop_without_subviews
            # spensor_global.loop_without_subviews.remove(constant_loop)
            new_result_type = result_type.spensor.element_type
            source_op_owner = op.source.owner
            assert isinstance(source_op_owner, Operation)

            parent_subview_ssa_list = getSSAListFromParentSubview(op)
            if len(parent_subview_ssa_list) != 0:
                cur_ssa_list: list[None | SSAValue] = [
                    None for _ in parent_subview_ssa_list
                ]
                use_loop_var_idx = 0
                for i, ssa_val in enumerate(parent_subview_ssa_list):
                    if ssa_val != zero:
                        cur_ssa_list[i] = parallel_block.args[use_loop_var_idx]
                        use_loop_var_idx += 1
                subview_ops = getSubViewOpWithSSA(
                    op.source, tuple(cur_ssa_list), source_type, new_result_type
                )
            else:
                subview_ops = getSubViewOp(op.source, op, source_type, new_result_type)

            parallel_block.add_ops(subview_ops)
            new_move_op = spensor.MoveOp(
                subview_ops[-1].results[0], result_type.get_memory()
            )
        else:
            new_move_op = spensor.MoveOp(source, result_type.get_memory())

        parallel_block.add_op(new_move_op)
        rewriter.replace_all_uses_with(op.result, new_move_op.result)
        rewriter.erase_matched_op()


class SpensorAddPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.AddOp, rewriter: PatternRewriter):
        parallel_block, _ = getParallelOp(op, rewriter)
        new_add_op = spensor.AddOp(op.lhs, op.rhs)
        parallel_block.add_op(new_add_op)
        rewriter.replace_all_uses_with(op.result, new_add_op.result)
        rewriter.erase_matched_op()


class SpensorMatmulPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MatmulOp, rewriter: PatternRewriter):
        parallel_block, _ = getParallelOp(op, rewriter)
        new_add_op = spensor.MatmulOp(op.lhs, op.rhs)
        parallel_block.add_op(new_add_op)
        rewriter.replace_all_uses_with(op.result, new_add_op.result)
        rewriter.erase_matched_op()


class SpensorMoveToPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MoveToOp, rewriter: PatternRewriter):
        parallel_block, _ = getParallelOp(op, rewriter)
        new_move_to_op = spensor.MoveToOp(op.source, op.destination)
        parallel_block.add_op(new_move_to_op)
        rewriter.erase_matched_op()


class SpensorReduceSumPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.ReduceSumOp, rewriter: PatternRewriter):
        parallel_block, _ = getParallelOp(op, rewriter)
        new_reduce_op = spensor.ReduceSumOp(op.operand)
        parallel_block.add_op(new_reduce_op)
        rewriter.replace_all_uses_with(op.result, new_reduce_op.result)
        rewriter.erase_matched_op()


class NDSpensorCombinePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.NDCombineOp, rewriter: PatternRewriter):
        source_type = op.source.type
        assert isinstance(source_type, spensor.SpensorType)

        result_nd_type = op.result.type
        assert isinstance(result_nd_type, spensor.NDSpensorType)
        result_type = result_nd_type.spensor.element_type

        parallel_block, _ = getParallelOp(op, rewriter)
        parallel_op = parallel_block.parent_op()
        assert parallel_op is not None

        inner_op = op.source.owner
        assert isinstance(inner_op, Operation)
        inner_parallel_op = inner_op.parent_op()
        # We can't use getParallelOp here because inner_op might be
        # newly generated
        assert isinstance(inner_parallel_op, scf.ParallelOp)
        assert isinstance(parallel_op, scf.ParallelOp)
        inner_parallel_block = inner_parallel_op.body.block

        memory_name = result_type.memory.memory_name
        memory = spensor_global.memory_mapping[memory_name]
        memory_type = spensor.MemoryType(memory_name, memory.shape)
        memory_op = spensor.GetMemoryOp(parallel_block.args, memory_name, memory_type)
        alloc_spensor_op = spensor.AllocSpensorOp(memory_op.result, result_type)
        addFront(alloc_spensor_op, parallel_op.body.block)
        rewriter.insert_op(memory_op, InsertPoint.before(alloc_spensor_op))
        # rewriter.insert_op([memory_op,alloc_spensor_op], InsertPoint.before(inner_parallel_op))

        subview_ops = getSubViewOp(
            alloc_spensor_op.result, op, result_type, source_type
        )
        move_op = spensor.MoveToOp(op.source, subview_ops[-1].results[0])
        inner_parallel_block.add_ops(subview_ops + [move_op])
        rewriter.replace_all_uses_with(op.result, alloc_spensor_op.result)
        rewriter.erase_matched_op()


class NDSpensorReducePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.NDReduceOp, rewriter: PatternRewriter):
        source_type = op.source.type
        assert isinstance(source_type, spensor.SpensorType)

        result_nd_type = op.result.type
        assert isinstance(result_nd_type, spensor.NDSpensorType)
        result_type = result_nd_type.spensor.element_type
        result_shape = result_type.element_type.get_shape()

        reduce_dim = toTupleInt(op.reduce_dim)
        nd_shape: list[int] = []
        parallel_block, _ = getParallelOp(op, rewriter)
        parallel_op = parallel_block.parent_op()
        assert isinstance(parallel_op, scf.ParallelOp)
        constant_loop = spensor_global.op_to_parent_loop[op]
        increase_bound_record = spensor_global.increase_bound_record[constant_loop]
        for dim in reduce_dim:
            for loop, idx in increase_bound_record:
                if idx == dim:
                    nd_shape.append(
                        getConstantFromSSA(
                            constant_loop_to_parallel_op[loop].upperBound[dim]
                        )
                    )
        reduce_nd_dim = tuple(nd_shape)
        new_alloc_shape = reduce_nd_dim + result_shape
        new_alloc_tensor_type = TensorType(
            result_type.element_type.element_type, new_alloc_shape
        )
        new_alloc_type = spensor.SpensorType(new_alloc_tensor_type, result_type.memory)

        inner_op = op.source.owner
        assert isinstance(inner_op, Operation)
        inner_parallel_op = inner_op.parent_op()
        # We can't use getParallelOp here because inner_op might be
        # newly generated
        assert isinstance(inner_parallel_op, scf.ParallelOp)
        inner_parallel_block = inner_parallel_op.body.block

        memory_name = result_type.memory.memory_name
        memory = spensor_global.memory_mapping[memory_name]
        memory_type = spensor.MemoryType(memory_name, memory.shape)
        memory_op = spensor.GetMemoryOp(parallel_block.args, memory_name, memory_type)
        alloc_spensor_op = spensor.AllocSpensorOp(memory_op.result, new_alloc_type)
        addFront(alloc_spensor_op, parallel_op.body.block)
        rewriter.insert_op(memory_op, InsertPoint.before(alloc_spensor_op))

        # In NDReduce, we didn't change the inner Spensor type, but just update the outer
        # NDSpensor shape
        ssa_list: list[SSAValue | None] = [None for _ in new_alloc_shape]
        for i, nd_dim in enumerate(reduce_dim):
            ssa_list[i] = inner_parallel_block.args[nd_dim]
        subview_ops = getSubViewOpWithSSA(
            alloc_spensor_op.result, tuple(ssa_list), new_alloc_type, result_type
        )
        move_op = spensor.MoveToOp(op.source, subview_ops[-1].results[0])
        inner_parallel_block.add_ops(subview_ops + [move_op])

        # At this step, all previous result has been write into alloc_spensor_op
        # Next, we start with a herd [1x1] to perform reduce operation
        result_spensor_op = spensor.AllocSpensorOp(memory_op.result, result_type)
        addFront(result_spensor_op, parallel_op.body.block)

        # Generate parallel op within for loops
        reduce_block = Block([], arg_types=[IndexType() for _ in result_shape])
        reduce_parallel_op = scf.ParallelOp(
            [zero for _ in result_shape],
            [one for _ in result_shape],
            [one for _ in result_shape],
            Region(reduce_block),
        )
        reduce_parallel_op.attributes = inner_parallel_op.attributes

        # First, we fill the op with zeros
        memory_name = source_type.memory.memory_name
        memory = spensor_global.memory_mapping[memory_name]
        memory_type = spensor.MemoryType(memory_name, memory.shape)
        inner_memory_op = spensor.GetMemoryOp(
            reduce_block.args, memory_name, memory_type
        )
        inner_result_spensor_type = spensor.SpensorType(
            result_spensor_op.result.type.element_type, memory_type
        )
        inner_result_spensor_op = spensor.AllocSpensorOp(
            inner_memory_op.result, inner_result_spensor_type
        )

        tensor_element_type = result_type.element_type.element_type
        assert isinstance(tensor_element_type, AnyFloat)
        zero_constant = arith.ConstantOp(
            FloatAttr(0, tensor_element_type), tensor_element_type
        )
        fill_op = spensor.FillOp(zero_constant.result, inner_result_spensor_op.result)
        reduce_block.add_ops(
            [inner_memory_op, inner_result_spensor_op, zero_constant, fill_op]
        )

        # Next, we expand for loops and perform the reduce operation
        ops, for_block, ind_vars = getNestedForLoop(reduce_nd_dim)
        reduce_block.add_ops(ops)
        reduce_ssa_list: list[SSAValue | None] = ind_vars + [None for _ in result_shape]
        reduce_subview_ops = getSubViewOpWithSSA(
            alloc_spensor_op.result, tuple(reduce_ssa_list), new_alloc_type, result_type
        )
        move_add_rhs_op = spensor.MoveOp(
            reduce_subview_ops[-1].results[0], source_type.memory
        )
        add_op = spensor.AddOp(
            inner_result_spensor_op.result, move_add_rhs_op.result, self_assign=True
        )
        for_block.add_ops(reduce_subview_ops + [move_add_rhs_op, add_op, scf.YieldOp()])

        # Finally, we move the result back
        move_result_op = spensor.MoveToOp(inner_result_spensor_op, result_spensor_op)
        reduce_block.add_op(move_result_op)
        rewriter.insert_op(reduce_parallel_op, InsertPoint.after(inner_parallel_op))

        rewriter.replace_all_uses_with(op.result, result_spensor_op.result)
        rewriter.erase_matched_op()


class NDSpensorRepeatPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.NDRepeatOp, rewriter: PatternRewriter):
        rewriter.replace_all_uses_with(op.result, op.source)
        rewriter.erase_matched_op()


class AppendReduceOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter):
        block = op.body.block
        block.add_op(scf.ReduceOp())


class ExpandParallel(ModulePass):
    name = "lower_ndspensor"

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
        constant_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([AppendReduceOpPattern()]),
            walk_reverse=False,
            apply_recursively=False,
        )
        constant_walker.rewrite_module(op)
