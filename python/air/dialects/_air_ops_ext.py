# ./python/air/dialects/_air_ops_ext.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..ir import *
from ._air_ops_gen import *
from . import arith
from ._ods_common import get_default_loc_context as _ods_get_default_loc_context
from ..extras.meta import region_op

from ..extras import types as T

class Launch(LaunchOp):
  """Specialization for LaunchOp class."""
  def __init__(self, name=None, sizes=[], async_token=None, async_dependencies=[], operands=[], attributes={}, loc=None, ip=None):
    iTy = IndexType.get()
    sizes = [arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)) for i in sizes]
    if name is not None:
      print(name)
      _ods_context = _ods_get_default_loc_context(loc)
    super().__init__(async_token=async_token,
                     async_dependencies=async_dependencies,
                     sizes=sizes, launch_operands=operands,
                     sym_name=name)
    operand_types = [s.type for s in sizes]*2 + \
                    [o.type for o in operands]
    self.regions[0].blocks.append(*operand_types)

class Segment(SegmentOp):
  """Specialization for SegmentOp class."""
  def __init__(self, name=None, sizes=[], async_token=None, async_dependencies=[], operands=[], attributes={}, loc=None, ip=None):
    iTy = IndexType.get()
    sizes = [arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)) for i in sizes]
    super().__init__(async_token=async_token,
                     async_dependencies=async_dependencies,
                     sizes=sizes, segment_operands=operands,
                     sym_name=name)
    operand_types = [s.type for s in sizes]*2 + \
                    [o.type for o in operands]
    self.regions[0].blocks.append(*operand_types)

class Herd(HerdOp):
  """Specialization for HerdOp class."""
  def __init__(self, name=None, sizes=[1,1], async_token=None, async_dependencies=[], operands=[], attributes={}, loc=None, ip=None):
    iTy = IndexType.get()
    sizes = [arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)) if isinstance(i, int) else i for i in sizes]
    super().__init__(async_token=async_token,
                     async_dependencies=async_dependencies,
                     sizes=sizes, herd_operands=operands,
                     sym_name=name)
    operand_types = [s.type for s in sizes]*2 + \
                    [o.type for o in operands]
    self.regions[0].blocks.append(*operand_types)

class ChannelGet(ChannelGetOp):
  def __init__(self, chan_name, indices, dst, dst_offsets=[], dst_sizes=[], dst_strides=[], async_token=None, async_dependencies=[], loc=None, ip=None):
    iTy = IndexType.get()
    indices = [arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)) if isinstance(i, int) else i for i in indices]
    super().__init__(async_token=async_token,
                     async_dependencies=async_dependencies, chan_name=chan_name,
                     indices=indices, dst=dst, dst_offsets=dst_offsets,
                     dst_sizes=dst_sizes, dst_strides=dst_strides, loc=loc, ip=ip)

class ChannelPut(ChannelPutOp):
  def __init__(self, chan_name, indices, src, src_offsets=[], src_sizes=[], src_strides=[], async_token=None, async_dependencies=[], loc=None, ip=None):
    iTy = IndexType.get()
    indices = [arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)) if isinstance(i, int) else i for i in indices]
    super().__init__(async_token=async_token,
                     async_dependencies=async_dependencies, chan_name=chan_name,
                     indices=indices, src=src, src_offsets=src_offsets,
                     src_sizes=src_sizes, src_strides=src_strides, loc=loc, ip=ip)

herd = region_op(Herd)
launch = region_op(Launch)
segment = region_op(Segment)
