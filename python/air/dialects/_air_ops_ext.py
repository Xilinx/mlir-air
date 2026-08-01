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


def split_static_dynamic(entries):
    """
    Split an access-pattern list into the (dynamic operands, static array) pair
    that the AIR memcpy ops store. A python int becomes a plain number in the
    static array; anything else stays an SSA operand, marked in the static array
    with ShapedType::kDynamic.
    """
    dynamic, static = [], []
    for e in entries:
        if isinstance(e, int):
            static.append(e)
        else:
            static.append(ShapedType.get_dynamic_size())
            dynamic.append(e)
    return dynamic, static


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
        for k, v in (attributes or {}).items():
            self.operation.attributes[k] = v
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
        for k, v in (attributes or {}).items():
            self.operation.attributes[k] = v
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
        for k, v in (attributes or {}).items():
            self.operation.attributes[k] = v
        operand_types = [s.type for s in sizes] * 2 + get_region_operand_types(operands)
        self.regions[0].blocks.append(*operand_types)


class Rank(RankOp):
    """Specialization for RankOp class."""

    def __init__(
        self,
        name=None,
        sizes=[],
        async_token=None,
        async_dependencies=[],
        operands=[],
        universe=None,
        attributes={},
        loc=None,
        ip=None,
    ):
        sizes = list(map(pyint_to_index, sizes))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            universe=universe,
            sizes=sizes,
            rank_operands=operands,
            sym_name=name,
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
        *,
        buffer_resources: Optional[Union[int, IntegerAttr]] = None,
    ):
        super().__init__(
            sym_name=sym_name,
            size=size,
            loc=loc,
            ip=ip,
        )

        # Optional objectFIFO depth knob: the `buffer_resources` attribute is
        # consumed by AIRToAIEPass (AIRToAIEPass.cpp, ChannelOp::getBufferResources)
        # as the depth of the lowered aie.objectfifo (default 1). Exposing it on the
        # placed Channel wrapper lets dataflow pin a deeper FIFO (e.g. depth-2 for
        # producer/consumer overlap) without hand-editing the IR.
        if buffer_resources is not None:
            if isinstance(buffer_resources, IntegerAttr):
                buffer_resources_attr = buffer_resources
            else:
                buffer_resources_attr = IntegerAttr.get(
                    IntegerType.get_signless(64), buffer_resources
                )
            super().attributes["buffer_resources"] = buffer_resources_attr

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
        pad_before=None,
        pad_after=None,
        loc=None,
        ip=None,
    ):
        if (pad_before is None) != (pad_after is None):
            raise ValueError(
                "pad_before and pad_after must both be specified or both omitted"
            )
        indices_typed = list(map(pyint_to_index, indices))
        dyn_offsets, static_offsets = split_static_dynamic(offsets)
        dyn_sizes, static_sizes = split_static_dynamic(sizes)
        dyn_strides, static_strides = split_static_dynamic(strides)
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            chan_name=chan_name,
            indices=indices_typed,
            dst=dst,
            dynamic_dst_offsets=dyn_offsets,
            dynamic_dst_sizes=dyn_sizes,
            dynamic_dst_strides=dyn_strides,
            static_dst_offsets=static_offsets,
            static_dst_sizes=static_sizes,
            static_dst_strides=static_strides,
            loc=loc,
            ip=ip,
        )
        # Set optional pad_before/pad_after attributes after construction,
        # since the generated __init__ doesn't accept them as kwargs.
        if pad_before is not None:
            self.operation.attributes["pad_before"] = DenseI32ArrayAttr.get(pad_before)
        if pad_after is not None:
            self.operation.attributes["pad_after"] = DenseI32ArrayAttr.get(pad_after)


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
        pad_before=None,
        pad_after=None,
        loc=None,
        ip=None,
    ):
        if (pad_before is None) != (pad_after is None):
            raise ValueError(
                "pad_before and pad_after must both be specified or both omitted"
            )
        indices_typed = list(map(pyint_to_index, indices))
        dyn_offsets, static_offsets = split_static_dynamic(offsets)
        dyn_sizes, static_sizes = split_static_dynamic(sizes)
        dyn_strides, static_strides = split_static_dynamic(strides)
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            chan_name=chan_name,
            indices=indices_typed,
            src=src,
            dynamic_src_offsets=dyn_offsets,
            dynamic_src_sizes=dyn_sizes,
            dynamic_src_strides=dyn_strides,
            static_src_offsets=static_offsets,
            static_src_sizes=static_sizes,
            static_src_strides=static_strides,
            loc=loc,
            ip=ip,
        )
        # Set optional pad_before/pad_after attributes after construction,
        # since the generated __init__ doesn't accept them as kwargs.
        if pad_before is not None:
            self.operation.attributes["pad_before"] = DenseI32ArrayAttr.get(pad_before)
        if pad_after is not None:
            self.operation.attributes["pad_after"] = DenseI32ArrayAttr.get(pad_after)


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
        pad_before=None,
        pad_after=None,
    ):
        if (pad_before is None) != (pad_after is None):
            raise ValueError(
                "pad_before and pad_after must both be specified or both omitted"
            )
        dyn_dst_offsets, static_dst_offsets = split_static_dynamic(dst_offsets)
        dyn_dst_sizes, static_dst_sizes = split_static_dynamic(dst_sizes)
        dyn_dst_strides, static_dst_strides = split_static_dynamic(dst_strides)

        dyn_src_offsets, static_src_offsets = split_static_dynamic(src_offsets)
        dyn_src_sizes, static_src_sizes = split_static_dynamic(src_sizes)
        dyn_src_strides, static_src_strides = split_static_dynamic(src_strides)

        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            dst=dst,
            dynamic_dst_offsets=dyn_dst_offsets,
            dynamic_dst_sizes=dyn_dst_sizes,
            dynamic_dst_strides=dyn_dst_strides,
            src=src,
            dynamic_src_offsets=dyn_src_offsets,
            dynamic_src_sizes=dyn_src_sizes,
            dynamic_src_strides=dyn_src_strides,
            static_dst_offsets=static_dst_offsets,
            static_dst_sizes=static_dst_sizes,
            static_dst_strides=static_dst_strides,
            static_src_offsets=static_src_offsets,
            static_src_sizes=static_src_sizes,
            static_src_strides=static_src_strides,
        )
        # Set optional pad_before/pad_after attributes after construction,
        # since the generated __init__ doesn't accept them as kwargs.
        if pad_before is not None:
            self.operation.attributes["pad_before"] = DenseI32ArrayAttr.get(pad_before)
        if pad_after is not None:
            self.operation.attributes["pad_after"] = DenseI32ArrayAttr.get(pad_after)


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
rank = region_op(Rank, terminator=lambda *_args: RankTerminatorOp())


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
