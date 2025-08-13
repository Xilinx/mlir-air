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
    Given a constant integer, returns an arith.ConstantOp with that value
    """
    const_op = arith.ConstantOp.from_int_and_width(val, IndexType())
    const_op.result.name_hint = "autogen_" + str(val) + "_index"
    return const_op


def getShape(shape: tuple[int, ...]) -> tuple[list[Operation], list[SSAValue]]:
    """
    Given a tensor shape, transfer it into a list of arith.Constant operations with their values
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
    Given a constant integer from arith.ConstantOp, fetch the integer
    """
    assert isinstance(val.owner, arith.ConstantOp)
    const = val.owner.value
    assert isinstance(const, IntegerAttr)
    const_val = const.value.data
    return const_val


def toTupleInt(array_attr: ArrayAttr[IntegerAttr]) -> tuple[int, ...]:
    """
    Given an ArrayAttr of IntegerArry, return the contained data as an int tuple
    """
    attr_list: list[IntegerAttr] = [attr for attr in array_attr]
    value_list = [attr.value.data for attr in attr_list]
    return tuple(value_list)


def toIntegerArrayAttr(int_list: Iterable[int | IntegerAttr]) -> ArrayAttr[IntegerAttr]:
    """
    Given an iterable of int or IntegerAttr, construct them as an arrayAttr
    """
    attr_list = [
        x if isinstance(x, IntegerAttr) else IntegerAttr.from_int_and_width(x, 64)
        for x in int_list
    ]
    return ArrayAttr(attr_list)


def getConstantOpByIndex(index: int) -> arith.ConstantOp:
    """
    Given an index, return the corresponding arith.ConstantOp
    This function contains a side effect on index_to_constant
    Must call by AppendConstant Pass to insert all operations
    """
    if index not in spensor_global.index_to_constant_op:
        spensor_global.index_to_constant_op[index] = getIndexConstant(index)
    return spensor_global.index_to_constant_op[index]


def getNestedForLoop(
    upper_bounds: tuple[int, ...]
) -> tuple[list[Operation], Block, list[SSAValue]]:
    """
    Given a list of upper bounds [2,3,4], construct nested scf.for loops as :
    for i 0 -> 2,
      for j 0 -> 3,
        for k 0 -> 4
          inner_block:

    and the result is (scf.for, inner_block, [i,j,k])
    return the outest for loop, the inner most block, and a list of induction variables

    Noite: This function calls getConstantOpByIndex
    Must call by AppendConstant Pass to insert all constant operations
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
    Replace all uses of an old SSA value by the provided one
    """
    for use in list(old_val.uses):
        use.operation.operands[use.index] = new_val


def copyToBlock(source: Block, dest: Block, new_arg_vals: list[SSAValue]):
    """
    Copy all operations from the source block into the dest block.
    Args in the source block is replaced by values in new_arg_vals
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
    Given an operation and a memory_tag
    Return its parent with the memory_tag specified
    This function is intended to find a scf.ParallelOp parent (not consider scf.for)
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
    Given an operation, insert it after the first op in the block or directly add it if no ops in the block
    """
    assert block is not None
    first_op = block.first_op
    if first_op is None:
        block.add_op(op)
    else:
        block.insert_op_before(op, first_op)


def getMemorySpace(op: Operation) -> Attribute | None:
    """
    Given an operatoin, find the nearest memory tag from its parent
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
    A helper function when creating bufferization.AllocTensorOp
    Add memory space attr by given the memory_space attr
    """
    tensor_op = bufferization.AllocTensorOp(result_type)
    if memory_space is not None:
        tensor_op.attributes["memory_space"] = memory_space
    return tensor_op
