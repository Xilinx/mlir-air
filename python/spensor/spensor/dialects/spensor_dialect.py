# ./python/spensor/spensor/dialects/spensor_dialect.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import abc
from typing import ClassVar, Iterable, Sequence
from math import prod


from xdsl.dialects.builtin import (
    MemRefType,
    ArrayAttr,
    StringAttr,
    IndexType,
    IntegerAttr,
    BoolAttr,
    TensorType,
    AnyFloat,
    AnySignlessIntegerOrIndexType,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    irdl_attr_definition,
    irdl_to_attr_constraint,
    operand_def,
    var_operand_def,
    result_def,
    attr_def,
    prop_def,
    AttrSizedOperandSegments,
)
from xdsl.utils.hints import isa
from spensor.utils.spensor_util import (
    toIntegerArrayAttr,
    toTupleInt,
    getConstantFromSSA,
)


@irdl_attr_definition
class MemoryType(ParametrizedAttribute, TypeAttribute):
    """
    Represents a memory type with a name and shape properties.

    For example,  L1 in AIE could have various shapes, such as [8, 4] or [4, 4]
    on Strix or Phoenix.

    Examples:
    - !spensor.memory<"L2", [6]> represents L2 with shape [6].
    - !spensor.memory<"L1", [4, 4]> represents L1 with shape [4, 4].
    """

    name = "spensor.memory"
    memory_name: StringAttr
    memory_shape: ArrayAttr[IntegerAttr]

    def get_memory_shape(self) -> tuple[int, ...]:
        return toTupleInt(self.memory_shape)

    def get_memory_name(self) -> str:
        return self.memory_name.data

    def __init__(
        self, memory_name: StringAttr | str, memory_shape: Iterable[int | IntegerAttr]
    ) -> None:
        if isinstance(memory_name, str):
            memory_name = StringAttr(memory_name)
        shape_list = [
            x if isinstance(x, IntegerAttr) else IntegerAttr.from_int_and_width(x, 64)
            for x in memory_shape
        ]
        shape_attr = ArrayAttr(shape_list)
        super().__init__(memory_name, shape_attr)


@irdl_attr_definition
class SpensorType(ParametrizedAttribute, TypeAttribute):
    """
    Represents a Spensor, which combines a tensor with its memory location,
    providing a way to specify where the tensor resides within a memory hierarchy.

    Examples:
    - !spensor.spensor<tensor<256x64xf32>, !spensor.memory<"L3", []>>
    """

    name = "spensor.spensor"
    element_type: TensorType
    memory: MemoryType

    def get_memory_shape(self) -> tuple[int, ...]:
        return self.memory.get_memory_shape()

    def get_memory_name(self) -> str:
        return self.memory.get_memory_name()

    def __init__(self, element_type: TensorType, memory: MemoryType) -> None:
        super().__init__(element_type, memory)


@irdl_attr_definition
class NDSpensorType(ParametrizedAttribute, TypeAttribute):
    """
    Represents an NDSpensor, which organizes multiple Spensors into a higher-dimensional tensor.

    Example:
    !spensor.ndspensor<
        tensor<4x
               !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>>>
    """

    name = "spensor.ndspensor"
    spensor: TensorType[SpensorType]

    def get_memory(self) -> MemoryType:
        return self.spensor.get_element_type().memory

    def get_memory_shape(self) -> tuple[int, ...]:
        return self.get_memory().get_memory_shape()

    def get_memory_name(self) -> str:
        return self.get_memory().get_memory_name()

    def __init__(self, spensor: TensorType[SpensorType]) -> None:
        super().__init__(spensor)


@irdl_op_definition
class SplitOp(IRDLOperation):
    """
    Splits a source Spensor into multiple partitions along a specified dimension.

    Example:
    Given a Spensor:
    - !spensor.spensor<tensor<256x64xf32>, !spensor.memory<"L3", []>>
      num_paritions = 4
      dim = 0

    Result:
    !spensor.ndspensor<
        tensor<4x
               !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>>>
    """

    name = "spensor.split"
    source = operand_def(SpensorType)
    num_partitions = operand_def(IndexType)
    dim = operand_def(IndexType)
    result = result_def(NDSpensorType)

    def get_result_type(
        self, source_type: Attribute, num_partitions: SSAValue, dim: SSAValue
    ) -> NDSpensorType:
        assert isinstance(source_type, SpensorType)
        shape = list(source_type.element_type.get_shape())
        dim_val = getConstantFromSSA(dim)
        num_partitions_val = getConstantFromSSA(num_partitions)
        shape[dim_val] //= num_partitions_val
        new_spensor_type = SpensorType(
            TensorType(source_type.element_type.element_type, shape), source_type.memory
        )
        new_ndspensor_type = NDSpensorType(
            TensorType(new_spensor_type, [num_partitions_val])
        )
        return new_ndspensor_type

    def __init__(
        self,
        source: SSAValue | Operation,
        num_partitions: SSAValue | Operation,
        dim: SSAValue | Operation,
    ):
        if isinstance(source, Operation):
            source = source.results[0]
        if isinstance(num_partitions, Operation):
            num_partitions = num_partitions.results[0]
        if isinstance(dim, Operation):
            dim = dim.results[0]
        result_type = self.get_result_type(source.type, num_partitions, dim)
        super().__init__(
            operands=[source, num_partitions, dim], result_types=[result_type]
        )


@irdl_op_definition
class NDBroadcastOp(IRDLOperation):
    """
    Broadcasts a Spensor or NDSpensor with the desired dimensions

    Reference:
    - NumPy broadcasting: https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
    """

    name = "ndspensor.broadcast"
    source = operand_def(SpensorType | NDSpensorType)
    dimensions = prop_def(ArrayAttr[IntegerAttr])
    result = result_def(NDSpensorType)

    def __init__(
        self, source: SSAValue | Operation, dimensions: Iterable[int | IntegerAttr]
    ):
        super().__init__(
            operands=[source], properties={"dimensions": toIntegerArrayAttr(dimensions)}
        )


@irdl_op_definition
class NDRepeatOp(IRDLOperation):
    """
    Repeats a Spensor or NDSpensor on the specified dimension with number of repetitions

    Reference:
    - NumPy Repeat https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
    """

    name = "ndspensor.repeat"
    source = operand_def(SpensorType | NDSpensorType)
    repeats = operand_def(IndexType)
    dim = operand_def(IndexType)
    result = result_def(NDSpensorType)

    def get_result_type(
        self,
        source_type: Attribute,
        repeats: SSAValue,
        dim: SSAValue,
        result_shape: Iterable[int],
    ) -> NDSpensorType:
        if isinstance(source_type, SpensorType):
            shape = [1]
            dim_val = getConstantFromSSA(dim)
            repeats_val = getConstantFromSSA(repeats)
            shape[dim_val] *= repeats_val
            assert prod(shape) == prod(result_shape)
            new_ndspensor_type = NDSpensorType(TensorType(source_type, result_shape))
            return new_ndspensor_type
        elif isinstance(source_type, NDSpensorType):
            spensor_type = source_type.spensor.element_type
            shape = list(source_type.spensor.get_shape())
            dim_val = getConstantFromSSA(dim)
            repeats_val = getConstantFromSSA(repeats)
            shape[dim_val] *= repeats_val
            assert prod(shape) == prod(result_shape)
            new_ndspensor_type = NDSpensorType(TensorType(spensor_type, result_shape))
            return new_ndspensor_type
        else:
            assert False and "Unsupport type"

    def __init__(
        self,
        source: SSAValue | Operation,
        repeats: SSAValue | Operation,
        dim: SSAValue | Operation,
        result_shape: Iterable[int],
    ):
        if isinstance(source, Operation):
            source = source.results[0]
        if isinstance(repeats, Operation):
            repeats = repeats.results[0]
        if isinstance(dim, Operation):
            dim = dim.results[0]
        result_type = self.get_result_type(source.type, repeats, dim, result_shape)
        super().__init__(operands=[source, repeats, dim], result_types=[result_type])


@irdl_op_definition
class NDCombineOp(IRDLOperation):
    """
    Combine a NDSpensor on a specified dimension into another NDSpensor on the specified memory

    Exmaple:
    - !spensor.ndspensor<
        tensor<4x2x
               !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L1", []>>>>
    - nd_dim = 1: Specifies the dimension to be combined in the NDSpensor shape.
    - dim = 0: Specifies the combining dimension in the inner Spensor type.
    - memory_name = "L2": Specifies the memory to use for the new NDSpensor.

    Result:
    - !spensor.ndspensor<
        tensor<4x1x
               !spensor.spensor<tensor<128x64xf32>, !spensor.memory<"L2", []>>>>
    """

    name = "ndspensor.combine"
    source = operand_def(NDSpensorType)
    nd_dim = operand_def(IndexType)
    dim = operand_def(IndexType)
    memory_name = prop_def(StringAttr)
    result = result_def(NDSpensorType)

    def get_result_type(
        self,
        source_type: Attribute,
        nd_dim: SSAValue,
        dim: SSAValue,
        memory: MemoryType,
    ) -> NDSpensorType:
        assert isinstance(source_type, NDSpensorType)
        nd_shape = list(source_type.spensor.get_shape())
        shape = list(source_type.spensor.element_type.element_type.get_shape())
        dim_val = getConstantFromSSA(dim)
        nd_dim_val = getConstantFromSSA(nd_dim)
        shape[dim_val] *= nd_shape[nd_dim_val]
        element_type = source_type.spensor.element_type.element_type.element_type
        new_spensor_type = SpensorType(TensorType(element_type, shape), memory)
        nd_shape.pop(nd_dim_val)
        if len(nd_shape) == 0:
            nd_shape = [1]
        new_ndspensor_type = NDSpensorType(TensorType(new_spensor_type, nd_shape))
        return new_ndspensor_type

    def __init__(
        self,
        source: SSAValue | Operation,
        nd_dim: SSAValue | Operation,
        dim: SSAValue | Operation,
        memory: MemoryType,
    ):
        if isinstance(source, Operation):
            source = source.results[0]
        if isinstance(nd_dim, Operation):
            nd_dim = nd_dim.results[0]
        if isinstance(dim, Operation):
            dim = dim.results[0]
        result_type = self.get_result_type(source.type, nd_dim, dim, memory)
        super().__init__(
            operands=[source, nd_dim, dim],
            properties={"memory_name": memory.memory_name},
            result_types=[result_type],
        )


@irdl_op_definition
class NDReduceOp(IRDLOperation):
    """
    Reduces a NDSpensor on given dimensions into another NDSpensor on the specified memory

    Exmaple:
    - !spensor.ndspensor<
        tensor<4x2x
               !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L1", []>>>>
    - reduce_dim = [0, 1]: Specifies dimensions to be reduced in the NDSpensor shape.
    - memory_name = "L2": Specifies the memory to use for the new NDSpensor.
    - op_name = "add": Specifies the reduce opereation

    Result:
    - !spensor.ndspensor<
        tensor<1x1x
               !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L2", []>>>>
    """

    name = "ndspensor.reduce"
    source = operand_def(NDSpensorType)
    reduce_dim = prop_def(ArrayAttr[IntegerAttr])
    memory_name = prop_def(StringAttr)
    op_name = prop_def(StringAttr)
    result = result_def(NDSpensorType)

    def get_result_type(
        self,
        source_type: Attribute,
        reduce_dim: Iterable[int | IntegerAttr],
        memory: MemoryType,
    ) -> NDSpensorType:
        assert isinstance(source_type, NDSpensorType)
        nd_shape = list(source_type.spensor.get_shape())
        shape = source_type.spensor.element_type.element_type.get_shape()

        reduce_dim_val = [
            x.value.data if isinstance(x, IntegerAttr) else x for x in reduce_dim
        ]
        assert len(reduce_dim_val) <= len(nd_shape)
        sorted(reduce_dim_val)
        for dim in reduce_dim_val[::-1]:
            nd_shape.pop(dim)

        element_type = source_type.spensor.element_type.element_type.element_type
        new_spensor_type = SpensorType(TensorType(element_type, shape), memory)
        if len(nd_shape) == 0:
            nd_shape = [1]
        new_ndspensor_type = NDSpensorType(TensorType(new_spensor_type, nd_shape))
        return new_ndspensor_type

    def __init__(
        self,
        source: SSAValue | Operation,
        reduce_dim: Iterable[int | IntegerAttr],
        memory: MemoryType,
        op_name: str | StringAttr,
    ):
        if isinstance(source, Operation):
            source = source.results[0]
        if isinstance(op_name, str):
            op_name = StringAttr(op_name)
        result_type = self.get_result_type(source.type, reduce_dim, memory)
        super().__init__(
            operands=[source],
            properties={
                "reduce_dim": toIntegerArrayAttr(reduce_dim),
                "memory_name": memory.memory_name,
                "op_name": op_name,
            },
            result_types=[result_type],
        )


@irdl_op_definition
class CombineToSpensorOp(IRDLOperation):
    """
    Combines a NDSpensor into a Spensor by merging dimensions based on the specified combine indices.

    Example:
    Given a NDSpensor:
    - !spensor.ndspensor<
        tensor<4x2x
               !spensor.spensor<tensor<64x64xf32>, !spensor.memory<"L3", []>>>>
    - combine_index = [1, 0]. This parameter must have the same length as ND shape ([4, 2]) here

    Result:
    - !spensor.spensor<tensor<128x256xf32>, !spensor.memory<"L3", []>>
    The first dimension [4] in NDSpensor is combined to [1] dimension in Spensor shape
    The second dimension [2] in NDSpensor is combined to [0] dimension in Spensor shape
    """

    name = "ndspensor.combine_to_spensor"
    source = operand_def(NDSpensorType)
    combine_index = prop_def(ArrayAttr[IntegerAttr])
    result = result_def(SpensorType)

    def get_result_type(
        self, source_type: Attribute, combine_index: ArrayAttr[IntegerAttr]
    ) -> SpensorType:
        combine_index_tuple = toTupleInt(combine_index)

        assert isinstance(source_type, NDSpensorType)
        nd_shape = source_type.spensor.get_shape()
        tensor_type = source_type.spensor.element_type.element_type
        tensor_shape = list(tensor_type.get_shape())
        assert len(nd_shape) == len(combine_index_tuple)
        for dim, index in enumerate(combine_index_tuple):
            tensor_shape[index] *= nd_shape[dim]

        new_spensor_type = SpensorType(
            TensorType(tensor_type.element_type, tensor_shape), source_type.get_memory()
        )

        return new_spensor_type

    def __init__(self, source: SSAValue, combine_index: tuple[int, ...]):
        int_attr_lists = [IntegerAttr.from_int_and_width(x, 64) for x in combine_index]
        combine_index_attr = ArrayAttr(int_attr_lists)
        result_type = self.get_result_type(source.type, combine_index_attr)
        super().__init__(
            operands=[source],
            properties={"combine_index": combine_index_attr},
            result_types=[result_type],
        )


@irdl_op_definition
class SplitAllOp(IRDLOperation):
    """
    Splits a Spensor along all dimensions into multiple partitions.

    Example:
    - !spensor.spensor<tensor<16x16xf32>, !spensor.memory<"L1", []>>
    - num_partition = [2, 4]: Specifies the number of partitions on every dimension

    Result:
    - !spensor.ndspensor<
        tensor<2x4x
               !spensor.spensor<tensor<8x4xf32>, !spensor.memory<"L1", []>>>>
    """

    name = "spensor.split_all"
    source = operand_def(SpensorType)
    num_partitions = prop_def(ArrayAttr[IntegerAttr])
    result = result_def(NDSpensorType)

    def get_result_type(
        self, source_type: Attribute, num_partitions: tuple[int, ...]
    ) -> NDSpensorType:
        assert isinstance(source_type, SpensorType)
        shape = source_type.element_type.get_shape()
        memory = source_type.memory
        assert len(shape) == len(num_partitions)
        new_shape = [i // j for i, j in zip(shape, num_partitions)]
        new_spensor_type = SpensorType(
            TensorType(source_type.element_type.element_type, new_shape), memory
        )
        new_nd_shape = [i for i in num_partitions if i != 1]
        if len(new_nd_shape) == 0:
            new_nd_shape = [1]
        return NDSpensorType(TensorType(new_spensor_type, new_nd_shape))

    def __init__(self, source: SSAValue, num_paritions: tuple[int, ...]):
        result_type = self.get_result_type(source.type, num_paritions)
        int_attr_lists = [IntegerAttr.from_int_and_width(x, 64) for x in num_paritions]
        array_attr = ArrayAttr(int_attr_lists)
        super().__init__(
            operands=[source],
            properties={"num_partitions": array_attr},
            result_types=[result_type],
        )


@irdl_op_definition
class GetMemoryOp(IRDLOperation):
    """
    Retrieves a memory by its name and possible indice on its shape

    Exmaple:
    memory = spensor.get_memory("L1", [0, 0])
    """

    name = "spensor.get_memory"
    indice = var_operand_def(IndexType)
    memory_name = prop_def(StringAttr)
    result = result_def(MemoryType)

    def __init__(
        self,
        indice: Sequence[SSAValue],
        memory_name: StringAttr,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[indice],
            properties={"memory_name": memory_name},
            result_types=[result_type],
        )


@irdl_op_definition
class AllocSpensorOp(IRDLOperation):
    """
    Allocs a Spensor from the given memory

    This operation should be used after retrieving a memory with GetMemoryOp to
    allocate a Spensor from that memory location.

    Exmaples:
    memory = spensor.get_memory("L1", [0, 0])
    !spensor<<2x2xf32>, "L1"> = spensor.alloc_spensor(memory)

    """

    name = "spensor.alloc_spensor"
    source = operand_def(MemoryType)
    result = result_def(SpensorType)

    def __init__(self, source: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[source], result_types=[result_type])


@irdl_op_definition
class DeclareMemoryOp(IRDLOperation):
    """
    Declares the connections and memory space of a specific memory.

    This operation is intended to be used at the beginning of a Module
    and is utilized in memory analysis.

    Note: This operation does not return any values.

    Examples:
    - spensor.declare_memory("L1", [4,4], ["L2", "L3"], ["L2", "L3"], 2)
    - spensor.declare_memory("L2", [6],  ["L3"], ["L3"], 1)
    - spensor.declare_memory("L3", [],  [], [])

    """

    name = "spensor.declare_memory"
    memory_name = prop_def(StringAttr)
    memory_shape = prop_def(ArrayAttr[IntegerAttr])
    load = prop_def(ArrayAttr[StringAttr])
    store = prop_def(ArrayAttr[StringAttr])
    memory_space = attr_def(Attribute)

    def __init__(
        self,
        memory_name: StringAttr | str,
        memory_shape: ArrayAttr[IntegerAttr] | Iterable[int | IntegerAttr],
        load: Iterable[str | StringAttr],
        store: Iterable[str | StringAttr],
        memory_space: int | IntegerAttr | None = None,
    ):
        if isinstance(memory_name, str):
            memory_name = StringAttr(memory_name)
        load_list: list[StringAttr] = [
            (x if isinstance(x, StringAttr) else StringAttr(x)) for x in load
        ]
        store_list: list[StringAttr] = [
            (x if isinstance(x, StringAttr) else StringAttr(x)) for x in store
        ]
        if isinstance(memory_shape, Iterable):
            memory_shape = toIntegerArrayAttr(memory_shape)
        if memory_space is not None:
            if isinstance(memory_space, int):
                memory_space = IntegerAttr.from_int_and_width(memory_space, 64)

        super().__init__(
            properties={
                "memory_name": memory_name,
                "memory_shape": memory_shape,
                "load": ArrayAttr(load_list),
                "store": ArrayAttr(store_list),
            },
            attributes={"memory_space": memory_space},
        )


@irdl_op_definition
class StoreSpensorOp(IRDLOperation):
    name = "spensor.store_spensor"
    source = operand_def(MemRefType)
    destination = operand_def(MemRefType)

    def __init__(self, source: SSAValue | Operation, destination: SSAValue | Operation):
        super().__init__(operands=[source, destination])


@irdl_op_definition
class MoveOp(IRDLOperation):
    """
    Moves the data from the source Spensor or NDSpensor to a new Spensor in
    the specified memory location.

    This operation is expected to be lowered to a combination of
    bufferization.alloc_tensor and linalg.copy operations.
    """

    name = "spensor.move"
    source = operand_def(NDSpensorType | SpensorType)
    memory_name = prop_def(StringAttr)
    result = result_def(NDSpensorType | SpensorType)

    def get_result_type(
        self, source_type: Attribute, memory: MemoryType
    ) -> NDSpensorType | SpensorType:
        if isinstance(source_type, NDSpensorType):
            spensor_type = source_type.spensor.element_type
            new_spensor_type = SpensorType(spensor_type.element_type, memory)
            return NDSpensorType(
                TensorType(new_spensor_type, source_type.spensor.get_shape())
            )
        else:
            assert isinstance(source_type, SpensorType)
            return SpensorType(source_type.element_type, memory)

    def __init__(self, source: SSAValue | Operation, memory: MemoryType):
        if isinstance(source, Operation):
            source = source.results[0]
        result_type = self.get_result_type(source.type, memory)
        super().__init__(
            operands=[source],
            properties={"memory_name": memory.memory_name},
            result_types=[result_type],
        )


@irdl_op_definition
class MoveToOp(IRDLOperation):
    """
    Moves the data from the source Spensor or NDSpensor to the destination.

    This operation is expected to be lowered to a linalg.copy
    """

    name = "spensor.move_to"
    source = operand_def(NDSpensorType | SpensorType)
    destination = operand_def(NDSpensorType | SpensorType)

    def __init__(self, source: SSAValue | Operation, destination: SSAValue | Operation):
        super().__init__(operands=[source, destination])


@irdl_op_definition
class SubviewOp(IRDLOperation):
    """
    Creates a subview of a Spensor based on the given offsets, sizes, and strides.

    Note: no static_offsets/sizes/strides attributes are provided here. Use arith.ConstantOp
    instead. They will be staticly determined in further lowerings to tensor.extract_slice.

    Reference:
    - Memref Subview: https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop
    """

    name = "spensor.subview"

    source = operand_def(SpensorType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    result = result_def(SpensorType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        source: SSAValue | Operation,
        offsets: Sequence[SSAValue],
        sizes: Sequence[SSAValue],
        strides: Sequence[SSAValue],
        result_type: Attribute,
    ):
        super().__init__(
            operands=[source, offsets, sizes, strides],
            result_types=[result_type],
        )


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    """
    Abstract base class for element-wise binary operations on tensors.

    This class is meant to be subclassed to define specific element-wise binary
    operations (e.g., addition, multiplication); thus lhs, rhs and result are expected
    to have the same type.

    Attributes:
    - self_assign (BoolAttr): Indicates whether the operation modifies the left-hand side operand in place.
    """

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(NDSpensorType | SpensorType)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    self_assign = attr_def(BoolAttr)

    result = result_def(T)

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        result_type: Attribute | None = None,
        self_assign: bool = False,
    ):
        if isinstance(lhs, Operation):
            lhs = lhs.results[0]
        if isinstance(rhs, Operation):
            rhs = rhs.results[0]
        if result_type is None:
            result_type = lhs.type

        self_assign_attr = BoolAttr.from_bool(self_assign)
        super().__init__(
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"self_assign": self_assign_attr},
        )


@irdl_op_definition
class AddOp(ElementwiseBinaryOperation):
    """
    Represents an element-wise addition operation between two spensors.
    """

    name = "spensor.add"


@irdl_op_definition
class ReduceSumOp(IRDLOperation):
    """
    Performs a reduction on the last dimension by summing its elements.

    Example:
    - !spensor.spensor<tensor<2x2xi8>, !spensor.memory<"L1", []>>
      with values [[1,2],[3,4]]

    Result:
    - !spensor.spensor<tensor<2x1xi8>, !spensor.memory<"L1", []>>
      with values [[3],[7]]
    """

    name = "spensor.reduce_sum"
    operand = operand_def(NDSpensorType | SpensorType | TensorType)
    result = result_def(NDSpensorType | SpensorType | TensorType)

    def get_result_type(
        self, source_type: Attribute
    ) -> NDSpensorType | SpensorType | TensorType:
        if isinstance(source_type, NDSpensorType):
            spensor = source_type.spensor.get_element_type()
            tensor_shape = list(spensor.element_type.get_shape())
            tensor_shape[-1] = 1
            new_spensor_type = SpensorType(
                TensorType(spensor.element_type.element_type, tensor_shape),
                spensor.memory,
            )
            new_nd_spensor_type = NDSpensorType(
                TensorType(new_spensor_type, source_type.spensor.get_shape())
            )
            return new_nd_spensor_type
        elif isinstance(source_type, SpensorType):
            tensor_shape = list(source_type.element_type.get_shape())
            tensor_shape[-1] = 1
            new_spensor_type = SpensorType(
                TensorType(source_type.element_type.element_type, tensor_shape),
                source_type.memory,
            )
            return new_spensor_type
        elif isa(source_type, TensorType):
            tensor_shape = list(source_type.get_shape())
            tensor_shape[-1] = 1
            new_memref_type = TensorType(source_type.get_element_type(), tensor_shape)
            return new_memref_type
        else:
            assert False and "Unspected type in reduceSumOp"

    def __init__(self, operand: SSAValue | Operation):
        if isinstance(operand, Operation):
            operand = operand.results[0]
        result_type = self.get_result_type(operand.type)
        super().__init__(operands=[operand], result_types=[result_type])


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """
    Represents a matrix multiplication operation.
    """

    name = "spensor.matmul"
    lhs = operand_def(NDSpensorType | SpensorType | TensorType)
    rhs = operand_def(NDSpensorType | SpensorType | TensorType)
    result = result_def(NDSpensorType | SpensorType | TensorType)

    def get_result_type(
        self, lhs_type: Attribute, rhs_type: Attribute
    ) -> NDSpensorType | SpensorType | TensorType:
        if isinstance(lhs_type, NDSpensorType) and isinstance(rhs_type, NDSpensorType):
            lhs_tensor_type = lhs_type.spensor.element_type.element_type
            rhs_tensor_type = rhs_type.spensor.element_type.element_type
            lhs_tensor_shape = lhs_tensor_type.get_shape()
            rhs_tensor_shape = rhs_tensor_type.get_shape()
            assert len(lhs_tensor_shape) == 2 and len(rhs_tensor_shape) == 2
            assert lhs_tensor_shape[1] == rhs_tensor_shape[0]
            result_tensor_shape = [lhs_tensor_shape[0], rhs_tensor_shape[1]]
            assert lhs_type.get_memory() == rhs_type.get_memory()
            result_spensor_type = SpensorType(
                TensorType(lhs_tensor_type.element_type, result_tensor_shape),
                lhs_type.get_memory(),
            )
            result_ndspensor_type = NDSpensorType(
                TensorType(result_spensor_type, lhs_type.spensor.get_shape())
            )
            return result_ndspensor_type
        elif isinstance(lhs_type, SpensorType) and isinstance(rhs_type, SpensorType):
            lhs_tensor_type = lhs_type.element_type
            rhs_tensor_type = rhs_type.element_type
            lhs_tensor_shape = lhs_tensor_type.get_shape()
            rhs_tensor_shape = rhs_tensor_type.get_shape()
            assert len(lhs_tensor_shape) == 2 and len(rhs_tensor_shape) == 2
            assert lhs_tensor_shape[1] == rhs_tensor_shape[0]
            result_tensor_shape = [lhs_tensor_shape[0], rhs_tensor_shape[1]]
            assert lhs_type.memory == rhs_type.memory
            result_spensor_type = SpensorType(
                TensorType(lhs_tensor_type.element_type, result_tensor_shape),
                lhs_type.memory,
            )
            return result_spensor_type
        elif isa(lhs_type, TensorType) and isa(rhs_type, TensorType):
            lhs_tensor_shape = lhs_type.get_shape()
            rhs_tensor_shape = rhs_type.get_shape()
            assert len(lhs_tensor_shape) == 2 and len(rhs_tensor_shape) == 2
            assert lhs_tensor_shape[1] == rhs_tensor_shape[0]
            result_tensor_shape = [lhs_tensor_shape[0], rhs_tensor_shape[1]]
            result_tensor_type = TensorType(
                lhs_type.get_element_type(), result_tensor_shape
            )
            return result_tensor_type
        else:
            assert False and "Unspected type in MatmulOp"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        if isinstance(lhs, Operation):
            lhs = lhs.results[0]
        if isinstance(rhs, Operation):
            rhs = rhs.results[0]
        result_type = self.get_result_type(lhs.type, rhs.type)
        super().__init__(operands=[lhs, rhs], result_types=[result_type])


@irdl_op_definition
class FillOp(IRDLOperation):
    """
    Fills the output tensor with a specified input value.

    Example:
    - inputs: 0x3f
    - outputs: !spensor.spensor<tensor<2x2xi8>, !spensor.memory<"L1", []>>

    Result:
    - !spensor.spensor<tensor<2x2xi8>, !spensor.memory<"L1", []>>
      with values [[0x3f, 0x3f],[0x3f, 0x3f]]
    """

    name = "spensor.fill"
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint((AnyFloat | AnySignlessIntegerOrIndexType))
    )
    inputs = operand_def(T)
    outputs = operand_def(NDSpensorType | SpensorType | TensorType)

    res = result_def(NDSpensorType | SpensorType | TensorType)

    def __init__(
        self,
        ins: SSAValue,
        outs: SSAValue,
    ):
        super().__init__(
            operands=[ins, outs],
            result_types=[outs.type],
        )


SpensorDialect = Dialect(
    "spensor",
    [
        MoveOp,
        MoveToOp,
        SplitOp,
        SplitAllOp,
        DeclareMemoryOp,
        GetMemoryOp,
        AllocSpensorOp,
        StoreSpensorOp,
        AddOp,
        ReduceSumOp,
        FillOp,
        MatmulOp,
    ],
    [MemoryType, SpensorType, NDSpensorType],
)

NDSpensorDialect = Dialect(
    "ndspensor",
    [
        NDBroadcastOp,
        NDCombineOp,
        CombineToSpensorOp,
        NDReduceOp,
        NDRepeatOp,
    ],
)
