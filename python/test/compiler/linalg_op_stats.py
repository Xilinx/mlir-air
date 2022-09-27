# ./python/test/compiler/linalg_op_stats.py -*- Python -*-

# Copyright (C) 2021=2022, Xilinx Inc.
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

# RUN: %PYTHON %s | FileCheck %s
from air.mlir.ir import *
from air.mlir.dialects import func
from air.mlir.dialects import linalg

from air.compiler.util import CostModel

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

# CHECK-LABEL: TEST: matmul_on_buffers_test
# CHECK:"matmul_on_buffers": {
# CHECK:  "linalg.matmul{{.*}}": {
# CHECK:    "arith.addf": 512,
# CHECK:    "arith.mulf": 512,
# CHECK:    "reads": 1536,
# CHECK:    "writes": 512
@run
def matmul_on_buffers_test():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @func.FuncOp.from_py_func(
        MemRefType.get((4, 16), f32), MemRefType.get((16, 8), f32),
        MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out])
    print(CostModel().op_stats(module))