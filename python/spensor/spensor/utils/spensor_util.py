# ./python/spensor/spensor/utils/spensor_util.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from typing import Iterable
from xdsl.dialects.builtin import (
    ArrayAttr,
    StringAttr,
    IndexType,
    IntegerAttr,
    TensorType,
)
from xdsl.dialects.builtin import IndexType, TensorType, StringAttr
from xdsl.ir import Operation, SSAValue, Block, Attribute
from xdsl.dialects import scf, arith, bufferization

import spensor.utils.spensor_global_analysis as spensor_global


def getIndexConstant(val: int) -> arith.ConstantOp:
    """
    Given a constant integer, returns an arith.ConstantOp with the specified value.
    This operation generates a constant of type IndexType.
    """
    const_op = arith.ConstantOp.from_int_and_width(val, IndexType())
    const_op.result.name_hint = "autogen_" + str(val) + "_index"
    return const_op


def getShape(shape: tuple[int, ...]) -> tuple[list[Operation], list[SSAValue]]:
    """
    Converts a given tensor shape tuple into a tuple of lists containing 
    arith.Constant operations and their corresponding SSA values.
    Each dimension of the shape is converted into a constant operation.
    """
    result_ops: list[Operation] = []
    result: list[SSAValue] = []
    for dim in shape:
        const_op = getIndexConstant(dim)
        result_ops.append(const_op)
        result.append(const_op.result)
    return result_ops, result


def getConstantFromSSA(val: SSAValue) -> int:
    """
    Retrieves the integer value from a given SSAValue that originates from an 
    arith.ConstantOp. Assumes the SSAValue owner is indeed a ConstantOp.
    """
    assert isinstance(val.owner, arith.ConstantOp)
    const = val.owner.value
    assert isinstance(const, IntegerAttr)
    const_val = const.value.data
    return const_val


def toTupleInt(array_attr: ArrayAttr[IntegerAttr]) -> tuple[int, ...]:
    """
    Converts an ArrayAttr of IntegerAttr elements into a tuple of integers.
    This function extracts the integer data from each IntegerAttr in the 
    provided ArrayAttr and returns it as a tuple of ints.
    """
    attr_list: list[IntegerAttr] = [attr for attr in array_attr]
    value_list = [attr.value.data for attr in attr_list]
    return tuple(value_list)


def toIntegerArrayAttr(int_list: Iterable[int | IntegerAttr]) -> ArrayAttr[IntegerAttr]:
    """
    Constructs an ArrayAttr of IntegerAttr elements from an iterable of ints 
    or IntegerAttr objects. Each int is converted to an IntegerAttr with 
    a width of 64 bits if necessary.
    """
    attr_list = [
        x if isinstance(x, IntegerAttr) else IntegerAttr.from_int_and_width(x, 64)
        for x in int_list
    ]
    return ArrayAttr(attr_list)


def getConstantOpByIndex(index: int) -> arith.ConstantOp:
    """
    Retrieves the arith.ConstantOp corresponding to the specified index.
    This function has a side effect on the index_to_constant_op global dictionary,
    ensuring the index is associated with an arith.ConstantOp. It should be 
    used in conjunction with the AppendConstant Pass to ensure all operations 
    are properly inserted.
    """
    if index not in spensor_global.index_to_constant_op:
        spensor_global.index_to_constant_op[index] = getIndexConstant(index)
    return spensor_global.index_to_constant_op[index]


def getNestedForLoop(
    upper_bounds: tuple[int, ...]
) -> tuple[list[Operation], Block, list[SSAValue]]:
    """
    Constructs nested scf.for loops given a list of upper bounds.
    
    For example, with upper_bounds = [2,3,4], the function will generate:
    scf.for i 0 -> 2,
      scf.for j 0 -> 3,
        scf.for k 0 -> 4
          inner_block:

    Returns a tuple containing:
    - The outermost scf.for loop operation.
    - The innermost block where operations can be inserted.
    - A list of induction variables corresponding to each loop level.

    Note: This function uses getConstantOpByIndex, thus AppendConstant Pass
    is needed at the end to ensure all constant operations are inserted.
    """
    assert len(upper_bounds) >= 1
    zero = getConstantOpByIndex(0)
    one = getConstantOpByIndex(1)

    cur_block = Block()
    cur_arg = cur_block.insert_arg(IndexType(), 0)
    first_ub = getConstantOpByIndex(upper_bounds[0])
    outer_for_op = scf.ForOp(zero, first_ub, one, [], cur_block)
    ind_vars: list[SSAValue] = [cur_arg]
    ops: list[Operation] = []

    for i in range(1, len(upper_bounds)):
        new_ub = getConstantOpByIndex(upper_bounds[i])
        new_block = Block()
        new_arg = new_block.insert_arg(IndexType(), 0)
        ind_vars.append(new_arg)

        new_for_op = scf.ForOp(zero, new_ub, one, [], new_block)
        cur_block.add_ops([new_for_op, scf.YieldOp()])
        cur_block = new_block

    ops.append(outer_for_op)
    assert cur_block is not None and outer_for_op is not None
    return ops, cur_block, ind_vars


def replaceUse(old_val: SSAValue, new_val: SSAValue):
    """
    Replaces all uses of an old SSA value by the provided one
    """
    for use in list(old_val.uses):
        use.operation.operands[use.index] = new_val


def copyToBlock(source: Block, dest: Block, new_arg_vals: list[SSAValue]):
    """
    Copies all operations from the source block into the destination block,
    replacing the source block's arguments with the specified new argument values.

    Preconditions:
    - The length of source.args must match the length of new_arg_vals.
    """
    assert len(source.args) == len(new_arg_vals)
    for old_arg, new_arg in zip(source.args, new_arg_vals):
        replaceUse(old_arg, new_arg)
    for source_op in source.ops:
        copy_op = source_op.clone()
        for old_res, new_res in zip(source_op.results, copy_op.results):
            replaceUse(old_res, new_res)
        dest.add_op(copy_op)


def findOpWithParentMemoryTag(
    op: Operation, memory_tag: StringAttr
) -> Operation | None:
    """
    Finds the parent operation with the specified memory_tag for a given operation,
    searching specifically for a parent scf.ParallelOp. scf.For operations are not considered.
    Note: It checks op's parent memory tag, not memory tag on op itself. 
    """
    cur_op = op
    parent_op = cur_op.parent_op()
    while parent_op is not None:
        if isinstance(parent_op, scf.ParallelOp):
            if parent_op.attributes.get("memory_tag") == memory_tag:
                return cur_op
        cur_op = parent_op
        parent_op = parent_op.parent_op()
    return None


def addFront(op: Operation, block: Block | None):
    """
    Inserts the given operation at the beginning of the specified block.
    If the block contains existing operations, the new operation is inserted 
    immediately after the first one; otherwise, it is added directly.
    """
    assert block is not None
    first_op = block.first_op
    if first_op is None:
        block.add_op(op)
    else:
        block.insert_op_before(op, first_op)


def getMemorySpace(op: Operation) -> Attribute | None:
    """
    Finds and returns the memory space attribute from the nearest parent 
    operation that contains a memory tag.
    """
    parent_op = op.parent_op()
    while parent_op is not None:
        if isinstance(parent_op, scf.ParallelOp):
            break
        parent_op = parent_op.parent_op()
    assert parent_op is not None
    memory_tag = parent_op.attributes["memory_tag"]
    if isinstance(memory_tag, StringAttr):
        return spensor_global.memory_mapping[memory_tag].memory_space
    return None


def getAllocTensorOpWithMemorySpace(
    result_type: TensorType, memory_space: Attribute | None
):
    """
    Creates a bufferization.AllocTensorOp with an optional memory space attribute.
    """
    tensor_op = bufferization.AllocTensorOp(result_type)
    if memory_space is not None:
        tensor_op.attributes["memory_space"] = memory_space
    return tensor_op
