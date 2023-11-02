# ./python/air/dialects/_air_ops_ext.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..ir import *
from ._air_ops_gen import *
from . import arith
from ._ods_common import get_default_loc_context as _ods_get_default_loc_context

class LaunchOp(LaunchOp):
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

class SegmentOp(SegmentOp):
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

class HerdOp(HerdOp):
  """Specialization for HerdOp class."""
  def __init__(self, name=None, sizes=[1,1], async_token=None, async_dependencies=[], operands=[], attributes={}, loc=None, ip=None):
    iTy = IndexType.get()
    sizes = [arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)) for i in sizes]
    super().__init__(async_token=async_token,
                     async_dependencies=async_dependencies,
                     sizes=sizes, herd_operands=operands,
                     sym_name=name)
    operand_types = [s.type for s in sizes]*2 + \
                    [o.type for o in operands]
    self.regions[0].blocks.append(*operand_types)