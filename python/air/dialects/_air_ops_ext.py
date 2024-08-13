# ./python/air/dialects/_air_ops_ext.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import functools
from typing import Optional, Sequence, Union

from ..ir import *
from ._air_ops_gen import *
from . import arith
from ._ods_common import get_default_loc_context as _ods_get_default_loc_context
from ..extras.meta import region_op

from ..extras import types as T
from .func import FuncOp, CallOp


def pyint_to_index(i):
    """
    Utility function to convert python int types to index types
    """
    return arith.ConstantOp.create_index(i) if isinstance(i, int) else i


def get_region_operand_types(operands):
    """
    Utility function to get the type of arguments given to region ops.
    """
    operand_types = []
    for o in operands:
        if isinstance(o, Value):
            operand_types.append(o.type)
        elif isinstance(o, OpView):
            if len(o.results.types) != 1:
                raise AttributeError(
                    f"Operation given to a region op as a parameter ({o}) has more "
                    "than one return type ({o.results.types}), which would lead to a mismatch "
                    "between number of operands and number of operand types"
                )
            operand_types += o.results.types
        else:
            raise AttributeError(
                f"Argument {o} is not a Value or an Operation: {type(o).mro()}"
            )
    return operand_types


class Launch(LaunchOp):
    """Specialization for LaunchOp class."""

    def __init__(
        self,
        name=None,
        sizes=[],
        async_token=None,
        async_dependencies=[],
        operands=[],
        attributes={},
        loc=None,
        ip=None,
    ):
        sizes = list(map(pyint_to_index, sizes))
        if name is not None:
            print(name)
            _ods_context = _ods_get_default_loc_context(loc)
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            sizes=sizes,
            launch_operands=operands,
            sym_name=name,
        )
        operand_types = [s.type for s in sizes] * 2 + get_region_operand_types(operands)
        self.regions[0].blocks.append(*operand_types)


class Segment(SegmentOp):
    """Specialization for SegmentOp class."""

    def __init__(
        self,
        name=None,
        sizes=[],
        async_token=None,
        async_dependencies=[],
        operands=[],
        attributes={},
        loc=None,
        ip=None,
    ):
        sizes = list(map(pyint_to_index, sizes))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            sizes=sizes,
            segment_operands=operands,
            sym_name=name,
        )
        operand_types = [s.type for s in sizes] * 2 + get_region_operand_types(operands)
        self.regions[0].blocks.append(*operand_types)


class Herd(HerdOp):
    """Specialization for HerdOp class."""

    def __init__(
        self,
        name=None,
        sizes=[1, 1],
        async_token=None,
        async_dependencies=[],
        operands=[],
        link_with=None,
        attributes={},
        loc=None,
        ip=None,
    ):
        sizes = list(map(pyint_to_index, sizes))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            sizes=sizes,
            herd_operands=operands,
            sym_name=name,
            link_with=link_with,
        )
        operand_types = [s.type for s in sizes] * 2 + get_region_operand_types(operands)
        self.regions[0].blocks.append(*operand_types)


class Channel(ChannelOp):
    def __init__(
        self,
        sym_name,
        broadcast_shape: Optional[
            Union[Sequence[Union[int, IntegerAttr, Operation, Value]], ArrayAttr]
        ] = None,
        size=None,
        loc=None,
        ip=None,
    ):
        super().__init__(
            sym_name=sym_name,
            size=size,
            loc=loc,
            ip=ip,
        )

        if not (broadcast_shape is None):
            static_sizes = []
            if isinstance(broadcast_shape, ArrayAttr):
                broadcast_shape_attr = broadcast_shape
            else:
                for size in broadcast_shape:
                    if isinstance(size, int):

                        static_sizes.append(IntegerAttr.get(T.index(), size))
                    else:
                        static_sizes.append(ShapedType.get_dynamic_size())
                broadcast_shape_attr = ArrayAttr.get(static_sizes)
            super().attributes["broadcast_shape"] = broadcast_shape_attr


class ChannelGet(ChannelGetOp):
    def __init__(
        self,
        chan_name,
        dst,
        offsets=[],
        sizes=[],
        strides=[],
        indices=[],
        async_token=None,
        async_dependencies=[],
        loc=None,
        ip=None,
    ):
        indices_typed = list(map(pyint_to_index, indices))
        dst_offsets_typed = list(map(pyint_to_index, offsets))
        dst_sizes_typed = list(map(pyint_to_index, sizes))
        dst_strides_typed = list(map(pyint_to_index, strides))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            chan_name=chan_name,
            indices=indices_typed,
            dst=dst,
            dst_offsets=dst_offsets_typed,
            dst_sizes=dst_sizes_typed,
            dst_strides=dst_strides_typed,
            loc=loc,
            ip=ip,
        )


class ChannelPut(ChannelPutOp):
    def __init__(
        self,
        chan_name,
        src,
        offsets=[],
        sizes=[],
        strides=[],
        indices=[],
        async_token=None,
        async_dependencies=[],
        loc=None,
        ip=None,
    ):
        indices_typed = list(map(pyint_to_index, indices))
        offsets_typed = list(map(pyint_to_index, offsets))
        sizes_typed = list(map(pyint_to_index, sizes))
        strides_typed = list(map(pyint_to_index, strides))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            chan_name=chan_name,
            indices=indices_typed,
            src=src,
            src_offsets=offsets_typed,
            src_sizes=sizes_typed,
            src_strides=strides_typed,
            loc=loc,
            ip=ip,
        )


class DmaMemcpyNd(DmaMemcpyNdOp):
    """Specialize DmaMemcpyNdOp class constructor to take python integers"""

    def __init__(
        self,
        dst,
        src,
        async_dependencies=[],
        async_token=None,
        dst_offsets=[],
        dst_sizes=[],
        dst_strides=[],
        src_offsets=[],
        src_sizes=[],
        src_strides=[],
    ):
        dst_offsets_typed = list(map(pyint_to_index, dst_offsets))
        dst_sizes_typed = list(map(pyint_to_index, dst_sizes))
        dst_strides_typed = list(map(pyint_to_index, dst_strides))

        src_offsets_typed = list(map(pyint_to_index, src_offsets))
        src_sizes_typed = list(map(pyint_to_index, src_sizes))
        src_strides_typed = list(map(pyint_to_index, src_strides))

        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            dst=dst,
            dst_offsets=dst_offsets_typed,
            dst_sizes=dst_sizes_typed,
            dst_strides=dst_strides_typed,
            src=src,
            src_offsets=src_offsets_typed,
            src_sizes=src_sizes_typed,
            src_strides=src_strides_typed,
        )


dma_memcpy_nd = DmaMemcpyNd


def module_builder(module_function):
    @functools.wraps(module_function)
    def module_builder_wrapper(*args, **kwargs):
        with Context() as ctx, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                module_function(*args, **kwargs)
        return module

    return module_builder_wrapper


herd = region_op(Herd, terminator=lambda *_args: HerdTerminatorOp())
launch = region_op(Launch, terminator=lambda *_args: LaunchTerminatorOp())
segment = region_op(Segment, terminator=lambda *_args: SegmentTerminatorOp())


def external_func(name, inputs, outputs=None, visibility="private"):
    if outputs is None:
        outputs = []
    return FuncOp(
        name=name, type=FunctionType.get(inputs, outputs), visibility=visibility
    )


# Wrapper for func CallOp.
class call(CallOp):
    """Specialize CallOp class constructor to take python integers"""

    def __init__(self, calleeOrResults, inputs=[], input_types=[]):
        attrInputs = []

        for i, itype in zip(inputs, input_types):
            if isinstance(i, int):
                attrInputs.append(arith.constant(itype, i))
            else:
                attrInputs.append(i)
        if isinstance(calleeOrResults, FuncOp):
            super().__init__(
                calleeOrResults=calleeOrResults,
                argumentsOrCallee=attrInputs,
            )
        else:
            super().__init__(
                calleeOrResults=input_types,
                argumentsOrCallee=FlatSymbolRefAttr.get(calleeOrResults),
                arguments=attrInputs,
            )
