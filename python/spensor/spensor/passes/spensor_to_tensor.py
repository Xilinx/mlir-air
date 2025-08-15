# ./python/spensor/spensor/passes/spensor_to_tensor.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from typing import Sequence
from xdsl.dialects.builtin import (
    FloatAttr,
    ModuleOp,
    IndexType,
    AnyFloat,
    MemRefType,
    TensorType,
    FunctionType,
    i64,
    DenseArrayBase,
)
from xdsl.ir import SSAValue, Attribute
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.dialects import (
    func,
    scf,
    memref,
    arith,
    linalg,
    tensor,
    bufferization,
)
import spensor.dialects.spensor_dialect as spensor
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.utils.hints import isa
from spensor.utils.spensor_util import (
    getMemorySpace,
    getAllocTensorOpWithMemorySpace,
    addFront,
    findOpWithParentMemoryTag,
    getConstantFromSSA,
    getNestedForLoop,
)
from xdsl.irdl import (
    Operand,
)

def lowerSpensorType(ty: spensor.SpensorType) -> TensorType:
    """
    Transfers a Spensor type to Tensor type
    """
    return ty.element_type


def lowerType(ty: Attribute) -> TensorType:
    """
    Transfers a general type to Tensor type
    """
    if isinstance(ty, spensor.SpensorType):
        return lowerSpensorType(ty)
    elif isa(ty, TensorType):
        return ty
    else:
        assert False and "Type not support"


def toMemrefType(ty: TensorType, memory_space: Attribute | None = None):
    """
    Transfers a TensorType to Memref Type with possible memory_space
    """
    if memory_space is None:
        return MemRefType(ty.element_type, ty.get_shape())
    return MemRefType(ty.element_type, ty.get_shape(), memory_space=memory_space)


class LowerFuncPattern(RewritePattern):
    """
    Replaces all arguments with Spensor Type to Tensor type in matched functions
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if isinstance(op.args[0].type, spensor.SpensorType):
            func_block = op.body.block
            args_len = len(op.args)
            for i in range(args_len):
                old_arg = op.args[i]
                new_arg_tensor_type = lowerType(old_arg.type)
                new_arg_type = toMemrefType(new_arg_tensor_type)
                new_arg = func_block.insert_arg(new_arg_type, i)
                to_tensor_op = bufferization.ToTensorOp(
                    new_arg, restrict=True, writable=True
                )
                addFront(to_tensor_op, func_block)
                old_arg.replace_by(to_tensor_op.tensor)
                func_block.erase_arg(old_arg)
            rewriter.has_done_action = True
            new_input_types = [arg.type for arg in op.args]
            new_function_type = FunctionType.from_lists(
                new_input_types, list(op.function_type.outputs)
            )
            op.function_type = new_function_type


class SpensorSubViewPattern(RewritePattern):
    """
    Lowers a SubviewOp to tensor.ExtractSliceOp
    
    Reference:
    - ExtractSliceOp https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract_slice-tensorextractsliceop
    """

    def get_subview_op(self,
        source: Operand,
        static_sizes: Sequence[int],
        static_offsets: Sequence[int] | None = None,
        static_strides: Sequence[int] | None = None,
        offsets: Sequence[Operand] | None = None,
        sizes: Sequence[Operand] | None = None,
        strides: Sequence[Operand] | None = None,
        result_type: Attribute | None = None,
    ) -> tensor.ExtractSliceOp:
        dims = len(static_sizes)
        offsets = [] if offsets is None else offsets
        sizes = [] if sizes is None else sizes
        strides = [] if strides is None else strides
        if not static_offsets:
            static_offsets = [memref.SubviewOp.DYNAMIC_INDEX] * len(offsets) + (
                [0] * (dims - len(offsets))
            )
        if not static_strides:
            static_strides = [memref.SubviewOp.DYNAMIC_INDEX] * len(strides) + (
                [1] * (dims - len(strides))
            )
        return tensor.ExtractSliceOp.build(
            operands=[
                source,
                offsets,
                sizes,
                strides,
            ],
            properties={
                "static_offsets": DenseArrayBase.from_list(
                    i64,
                    static_offsets,
                ),
                "static_sizes": DenseArrayBase.from_list(
                    i64,
                    static_sizes,
                ),
                "static_strides": DenseArrayBase.from_list(
                    i64,
                    static_strides,
                ),
            },
            result_types=[result_type],
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.SubviewOp, rewriter: PatternRewriter):
        result_type = lowerType(op.result.type)
        sizes = [getConstantFromSSA(val) for val in op.sizes]
        new_slice_op = self.get_subview_op(
            op.source, sizes, [], [], op.offsets, [], op.strides, result_type
        )
        rewriter.replace_matched_op(new_slice_op)


class SpensorMovePattern(RewritePattern):
    """
    Lowers a MoveOp to a combination of bufferization.alloc_tensor and linalg.CopyOp
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MoveOp, rewriter: PatternRewriter):
        result_type = op.result.type
        tensor_type = lowerType(result_type)

        op_with_parent_memory_tag = findOpWithParentMemoryTag(op, op.memory_name)
        assert op_with_parent_memory_tag is not None
        memory_space = getMemorySpace(op_with_parent_memory_tag)
        tensor_op = getAllocTensorOpWithMemorySpace(tensor_type, memory_space)

        copy_op = linalg.CopyOp([op.source], [tensor_op.tensor], [tensor_type])
        addFront(tensor_op, op_with_parent_memory_tag.parent_block())
        rewriter.insert_op_before_matched_op(copy_op)
        rewriter.replace_all_uses_with(op.result, tensor_op.tensor)
        rewriter.erase_matched_op()


class SpensorMoveToPattern(RewritePattern):
    """
    Lowers a MoveOp to a linalg.CopyOp
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MoveToOp, rewriter: PatternRewriter):
        source = op.source
        dest = op.destination
        tensor_type = lowerType(source.type)
        copy_op = linalg.CopyOp([source], [dest], [tensor_type])
        rewriter.insert_op_before_matched_op(copy_op)
        rewriter.erase_matched_op()


class SpensorAddPattern(RewritePattern):
    """
    Lowers a AddOp to a linalg.AddOp.
    If it's not a self-assignment add, create and store 
    the result to a bufferization.alloc_tensor
    """    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.AddOp, rewriter: PatternRewriter):
        lhs, rhs = op.lhs, op.rhs
        tensor_type = lowerType(op.result.type)

        if 0 == op.self_assign.value.data:
            memory_space = getMemorySpace(op)
            tensor_op = getAllocTensorOpWithMemorySpace(tensor_type, memory_space)
            addFront(tensor_op, op.parent_block())
            add_op = linalg.AddOp([lhs, rhs], [tensor_op.tensor], [tensor_type])
            rewriter.insert_op_before_matched_op(add_op)
            rewriter.replace_all_uses_with(op.result, tensor_op.tensor)
            rewriter.erase_matched_op()
        else:
            add_op = linalg.AddOp([lhs, rhs], [lhs], [tensor_type])
            rewriter.replace_matched_op(add_op)


class SpensorMatmulPattern(RewritePattern):
    """
    Lowers a MatmulOp to a linalg.MatmulOp.    
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.MatmulOp, rewriter: PatternRewriter):
        lhs, rhs = op.lhs, op.rhs
        tensor_type = lowerType(op.result.type)

        matmul_op = linalg.MatmulOp([lhs, rhs], [lhs], [tensor_type])
        rewriter.replace_matched_op(matmul_op)


class SpensorFillPattern(RewritePattern):
    """
    Lowers a FillOp to a linalg.FillOp.    
    """    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.FillOp, rewriter: PatternRewriter):
        constant = op.inputs
        tensor_type = lowerType(op.outputs.type)
        fill_op = linalg.FillOp([constant], [op.outputs], [tensor_type])
        rewriter.replace_matched_op(fill_op)


class SpensorReduceSumPattern(RewritePattern):
    """
    Manually implements a reduction operation on a spensor's last dimension.
    We could use linalg operation instead.
    Examples:
    - reduce_sum(<4x4xf32>) -> <4x1xf32>

    Result:
        result: <4x1xf32>
        fill(0, result)
        scf.for i: 0 -> 4
          scf.for j: 0 -> 4
            tmp = load(result, i, 0)
            tmp += load(source, i, j)
            store(tmp, reuslt, i, 0)
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.ReduceSumOp, rewriter: PatternRewriter):
        source_type = lowerType(op.operand.type)
        result_type = lowerType(op.result.type)

        memory_space = getMemorySpace(op)
        tensor_op = getAllocTensorOpWithMemorySpace(result_type, memory_space)
        addFront(tensor_op, op.parent_block())
        element_type = result_type.element_type
        assert isinstance(element_type, AnyFloat)
        zero_consant_attr = FloatAttr(0, element_type)
        zero_constant = arith.ConstantOp(zero_consant_attr, result_type.element_type)
        new_fill_op = linalg.FillOp(
            [zero_constant.result], [tensor_op.tensor], [result_type]
        )
        rewriter.insert_op_before_matched_op([zero_constant, new_fill_op])

        memory_space = getMemorySpace(op)
        store_to_memref_op = bufferization.ToMemRefOp(
            operands=[tensor_op.tensor],
            result_types=[toMemrefType(result_type, memory_space)],
        )
        load_to_memref_op = bufferization.ToMemRefOp(
            operands=[op.operand],
            result_types=[toMemrefType(source_type, memory_space)],
        )
        rewriter.insert_op_before_matched_op([store_to_memref_op, load_to_memref_op])

        ops, for_block, ind_vars = getNestedForLoop(source_type.get_shape())
        rewriter.insert_op_before_matched_op(ops)

        load_source = memref.LoadOp.get(load_to_memref_op, ind_vars)
        zero_constant = arith.ConstantOp.from_int_and_width(0, IndexType())
        dest_access: list[SSAValue] = ind_vars
        dest_access[-1] = zero_constant.result
        load_dest = memref.LoadOp.get(store_to_memref_op, dest_access)
        add_op = arith.AddfOp(load_source.res, load_dest.res)
        store_result = memref.StoreOp.get(
            add_op.result, store_to_memref_op, dest_access
        )
        for_block.add_ops([load_source, zero_constant, load_dest, add_op, store_result])
        for_block.add_op(scf.YieldOp())

        rewriter.replace_all_uses_with(op.result, tensor_op.tensor)
        rewriter.erase_matched_op()


class SpensorAllocSpensorPattern(RewritePattern):
    """
    Lowers a AllocSpensorOp to a bufferization.alloc_tensor
    """    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.AllocSpensorOp, rewriter: PatternRewriter):
        result_type = lowerType(op.result.type)
        memory_space = getMemorySpace(op)
        tensor_op = getAllocTensorOpWithMemorySpace(result_type, memory_space)
        addFront(tensor_op, op.parent_block())
        rewriter.replace_all_uses_with(op.result, tensor_op.tensor)
        rewriter.erase_matched_op()


class SpensorToTensor(ModulePass):
    name = "spensor_to_tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerFuncPattern(),
                    SpensorSubViewPattern(),
                    SpensorMovePattern(),
                    SpensorFillPattern(),
                    SpensorMoveToPattern(),
                    SpensorAddPattern(),
                    SpensorMatmulPattern(),
                    SpensorReduceSumPattern(),
                    SpensorAllocSpensorPattern(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        walker.rewrite_module(op)
