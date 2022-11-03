# ./python/air/dialects/_air_ops_ext.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

try:
  from ..mlir.ir import *
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

class AirHierarchyOp:
  def __init__(self, name=None, sizes=[1,1], results=[], async_deps=[], operands=[], attributes={}, loc=None, ip=None):
    if name:
      attributes["sym_name"] = StringAttr.get(str(name))
    super().__init__(self.build_generic(
      results=results,
      operands=[async_deps,sizes,operands],
      attributes=attributes,
      loc=loc,
      ip=ip))
    operand_types = [s.type for s in sizes]*2 + \
                    [o.type for o in operands]
    self.regions[0].blocks.append(*operand_types)

  @property
  def body(self):
    return self.regions[0].blocks[0]

  @property
  def name(self) -> StringAttr:
    return StringAttr(self.attributes["sym_name"])

class LaunchOp(AirHierarchyOp):
  """Specialization for LaunchOp class."""
  def __init__(self, name=None, sizes=[], results=[], async_deps=[], operands=[], attributes={}, loc=None, ip=None):
    super().__init__(name, sizes, results, async_deps, operands, attributes, loc, ip)

class PartitionOp(AirHierarchyOp):
  """Specialization for PartitionOp class."""
  def __init__(self, name=None, sizes=[], results=[], async_deps=[], operands=[], attributes={}, loc=None, ip=None):
    super().__init__(name, sizes, results, async_deps, operands, attributes, loc, ip)

class HerdOp(AirHierarchyOp):
  """Specialization for HerdOp class."""
  def __init__(self, name=None, sizes=[1,1], results=[], async_deps=[], operands=[], attributes={}, loc=None, ip=None):
    super().__init__(name, sizes, results, async_deps, operands, attributes, loc, ip)
