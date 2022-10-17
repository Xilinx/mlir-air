# ./python/air/dialects/_air_ops_ext.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
