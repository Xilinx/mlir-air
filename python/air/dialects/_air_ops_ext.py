# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

try:
  from ..ir import *
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

class HerdLaunchOp:
  """Specialization for herd_launch op class."""

  def __init__(self, name, size_x: Value, size_y: Value, operands, async_deps=[], loc=None, ip=None):
    results = []
    attributes = {}
    attributes["sym_name"] = StringAttr.get(str(name))
    super().__init__(self.build_generic(
      results=results,
      operands=[async_deps,size_x,size_y,operands],
      attributes=attributes,
      loc=loc,
      ip=ip))
    operand_types = [IndexType.get()]*4 + \
                    [o.type for o in operands]
    self.regions[0].blocks.append(*operand_types)

  @property
  def body(self):
    return self.regions[0].blocks[0]

  @property
  def name(self) -> StringAttr:
    return StringAttr(self.attributes["sym_name"])
