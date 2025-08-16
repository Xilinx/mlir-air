from xdsl.dialects.builtin import (
    ModuleOp,
    FunctionType,
)
from xdsl.ir import Operation, SSAValue
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.dialects import func
import spensor.dialects.spensor_dialect as spensor
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)

from spensor.dialects.spensor_dialect import toTupleInt


def reverseCombine(
    combineOp: spensor.CombineToSpensorOp, moveOp: spensor.MoveOp, output: SSAValue
) -> list[Operation]:
    result_type = combineOp.result.type
    result_shape = result_type.element_type.get_shape()
    combine_index = toTupleInt(combineOp.combine_index)
    source_type = combineOp.source.type
    assert isinstance(source_type, spensor.NDSpensorType)
    nd_shape = source_type.spensor.get_shape()
    if result_shape == nd_shape:
        return [spensor.MoveToOp(combineOp.source, output)]
    split_args = [1 for _ in result_shape]
    assert len(combine_index) == len(nd_shape)
    for num_partition, index in zip(nd_shape, combine_index):
        split_args[index] = num_partition
    split_all_op = spensor.SplitAllOp(output, tuple(split_args))
    move_to_op = spensor.MoveToOp(moveOp.source, split_all_op.result)
    return [split_all_op, move_to_op]


class BufferizeOutputPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        block = op.body.block

        last_op = block.last_op
        assert last_op is not None and isinstance(last_op, func.ReturnOp)
        return_vals = last_op.operands
        if len(return_vals) == 1:
            return_val = return_vals[0]
            output_arg = block.insert_arg(return_val.type, len(block.args))
            combine_op = return_val.owner
            assert isinstance(combine_op, spensor.CombineToSpensorOp)

            move_op = combine_op.source.owner
            assert isinstance(move_op, spensor.MoveOp)
            split_ops = reverseCombine(combine_op, move_op, output_arg)
            block.insert_ops_before(split_ops + [func.ReturnOp()], last_op)

            rewriter.erase_op(last_op)
            rewriter.erase_op(combine_op)
            rewriter.erase_op(move_op)

            input_types = list(op.function_type.inputs.data) + [output_arg.type]
            new_function_type = FunctionType.from_lists(input_types, [])
            op.function_type = new_function_type


class BufferizeOutput(ModulePass):
    name = "lower_ndspensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([BufferizeOutputPattern()]),
            walk_reverse=False,
            apply_recursively=False,
        )
        walker.rewrite_module(op)
