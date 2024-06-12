# ./python/air/dialects/_air_ops_ext.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import functools

from ..ir import *
from ._air_ops_gen import *
from . import arith
from ._ods_common import get_default_loc_context as _ods_get_default_loc_context
from ..extras.meta import region_op

from air.dialects.memref import AllocOp, DeallocOp
from ..extras import types as T


def int_to_index(i):
    """
    Utility function to convert python int types to index types
    """
    return arith.ConstantOp.create_index(i) if isinstance(i, int) else i


class Alloc(AllocOp):
    """Specialization for AllocOp class."""

    def __init__(
        self,
        memref,
        dynamicSizes=[],
        symbolOperands=[],
        *args,
        **kwargs,
    ):
        super().__init__(memref, dynamicSizes, symbolOperands, *args, **kwargs)


# This is included to keep Alloc/Dealloc symmetrically named operations
Dealloc = DeallocOp


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
        sizes = list(map(int_to_index, sizes))
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
        operand_types = [s.type for s in sizes] * 2 + [o.type for o in operands]
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
        sizes = list(map(int_to_index, sizes))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            sizes=sizes,
            segment_operands=operands,
            sym_name=name,
        )
        operand_types = [s.type for s in sizes] * 2 + [o.type for o in operands]
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
        attributes={},
        loc=None,
        ip=None,
    ):
        sizes = list(map(int_to_index, sizes))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            sizes=sizes,
            herd_operands=operands,
            sym_name=name,
        )
        operand_types = [s.type for s in sizes] * 2 + [o.type for o in operands]
        self.regions[0].blocks.append(*operand_types)


class ChannelGet(ChannelGetOp):
    def __init__(
        self,
        chan_name,
        dst,
        dst_offsets=[],
        dst_sizes=[],
        dst_strides=[],
        indices=[],
        async_token=None,
        async_dependencies=[],
        loc=None,
        ip=None,
    ):
        indices_typed = list(map(int_to_index, indices))
        dst_offsets_typed = list(map(int_to_index, dst_offsets))
        dst_sizes_typed = list(map(int_to_index, dst_sizes))
        dst_strides_typed = list(map(int_to_index, dst_strides))
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
        src_offsets=[],  # Try w/ 0,0 first (should be first block)
        src_sizes=[],
        src_strides=[],  # Tile size []
        indices=[],  # Use the channel to describe
        async_token=None,
        async_dependencies=[],
        loc=None,
        ip=None,
    ):
        indices_typed = list(map(int_to_index, indices))
        src_offsets_typed = list(map(int_to_index, src_offsets))
        src_sizes_typed = list(map(int_to_index, src_sizes))
        src_strides_typed = list(map(int_to_index, src_strides))
        super().__init__(
            async_token=async_token,
            async_dependencies=async_dependencies,
            chan_name=chan_name,
            indices=indices_typed,
            src=src,
            src_offsets=src_offsets_typed,
            src_sizes=src_sizes_typed,
            src_strides=src_strides_typed,
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
        dst_offsets_typed = list(map(int_to_index, dst_offsets))
        dst_sizes_typed = list(map(int_to_index, dst_sizes))
        dst_strides_typed = list(map(int_to_index, dst_strides))

        src_offsets_typed = list(map(int_to_index, src_offsets))
        src_sizes_typed = list(map(int_to_index, src_sizes))
        src_strides_typed = list(map(int_to_index, src_strides))

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


herd = region_op(Herd)
launch = region_op(Launch)
segment = region_op(Segment)
