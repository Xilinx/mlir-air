from xdsl.dialects.builtin import (
    ModuleOp,
    IndexType,
    StringAttr,
    AffineMapAttr,
)
from xdsl.ir import Operation, SSAValue, Block, Attribute, BlockArgument, Region, Use
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.dialects import func, scf, memref, arith, affine
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

from spensor.dialects.spensor_dialect import (
    AllocSpensorOp,
    getConstantFromSSA,
    toTupleInt,
)
from spensor.utils.spensor_util import (
    copyToBlock,
    getConstantOpByIndex,
)
from spensor.utils.spensor_global_analysis import Memory
import spensor.utils.spensor_global_analysis as spensor_global


zero_op = getConstantOpByIndex(0)
one_op = getConstantOpByIndex(1)
zero = zero_op.result
one = one_op.result


def getLinearizeIndexAffineMap(shape: tuple[int, ...]) -> AffineMap:
    key = "lambda "
    args = ["x" + str(i) for i in range(len(shape))]
    factors = [1]
    for dim in shape[:0:-1]:
        factors.append(dim * factors[-1])
    factors.reverse()
    terms = [arg + "*" + str(fac) for arg, fac in zip(args, factors)]
    lambda_expr = key + ",".join(args) + ": (" + "+".join(terms) + ",)"
    result = AffineMap.from_callable(eval(lambda_expr), dim_symbol_split=(0, len(args)))
    return result


class TileParallelPattern(RewritePattern):
    def tile_parallel_op(
        self, upper_bounds: tuple[int, ...], memory_shape: tuple[int, ...]
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        bounds_queue = list(upper_bounds)
        shape_queue = list(memory_shape)
        tile_all_bounds: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        while len(bounds_queue) > 0:
            bound = bounds_queue.pop(0)
            if len(shape_queue) == 0:
                tile_all_bounds.append(((), (bound,)))
            else:
                extra_tile_current_bound = ()
                if shape_queue[0] >= bound:
                    shape_queue.pop(0)
                    tile_current_bound = [bound]
                else:
                    tile_current_bound: list[int] = []
                    while len(shape_queue) > 0:
                        shape = shape_queue.pop(0)
                        for try_shape in range(shape, 0, -1):
                            if bound % try_shape == 0:
                                tile_current_bound.append(try_shape)
                                bound //= try_shape
                                break
                        # One dimension for each dimension
                        break
                        # Multi dimension for one dimension
                        if bound == 1:
                            break
                    if bound != 1:
                        extra_tile_current_bound = (bound,)
                tile_all_bounds.append(
                    (tuple(tile_current_bound), extra_tile_current_bound)
                )
        return tuple(tile_all_bounds)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter):
        if "memory_tag" in op.attributes:
            memory_tag = op.attributes["memory_tag"]
            assert isinstance(memory_tag, StringAttr)
            memory = spensor_global.memory_mapping[memory_tag]
            if len(memory.shape) > 0:
                upper_bounds: tuple[int, ...] = tuple(
                    [getConstantFromSSA(x) for x in op.upperBound]
                )
                memory_shape = toTupleInt(memory.shape)
                tile_parallel_bounds = self.tile_parallel_op(upper_bounds, memory_shape)
                ub_ops: list[arith.ConstantOp] = []
                for memory_tile, _ in tile_parallel_bounds:
                    ub_ops += [getConstantOpByIndex(idx) for idx in memory_tile]
                lb_ops = [zero_op for _ in ub_ops]
                step_ops = [one_op for _ in ub_ops]
                block = Block([], arg_types=[IndexType() for _ in ub_ops])
                source_block = op.body.block
                parallel_op = scf.ParallelOp(lb_ops, ub_ops, step_ops, Region(block))
                parallel_op.attributes = op.attributes
                result_op: scf.ParallelOp | scf.ForOp = parallel_op

                new_arg_vals: list[SSAValue] = []
                memory_tile_args = list(block.args)

                for i, (memory_tile, extra_tile) in enumerate(tile_parallel_bounds):
                    index_list = memory_tile_args[0 : len(memory_tile)]
                    memory_tile_args = memory_tile_args[len(memory_tile) :]
                    index_bound_list = memory_tile
                    if len(extra_tile) != 0:
                        assert len(extra_tile) == 1
                        for_bound = extra_tile[0]
                        for_block = Block(
                            [result_op, scf.YieldOp()], arg_types=[IndexType()]
                        )
                        for_op = scf.ForOp(
                            zero, getConstantOpByIndex(for_bound), one, [], for_block
                        )
                        result_op = for_op
                        index_list += list(for_block.args)
                        index_bound_list += extra_tile
                    sum_affine_map = getLinearizeIndexAffineMap(index_bound_list)
                    affine_op = affine.ApplyOp(
                        index_list, AffineMapAttr(sum_affine_map)
                    )
                    block.add_op(affine_op)
                    new_arg_vals.append(affine_op.result)
                assert len(new_arg_vals) == len(source_block.args)
                copyToBlock(source_block, block, new_arg_vals)

                rewriter.insert_op_before_matched_op(result_op)
                rewriter.erase_matched_op()
                # rewriter.replace_matched_op(result_op)


class TileParallel(ModulePass):
    name = "memory_analysis"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([TileParallelPattern()]),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
