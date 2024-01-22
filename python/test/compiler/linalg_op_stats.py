# ./python/test/compiler/linalg_op_stats.py -*- Python -*-

# Copyright (C) 2021=2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s
from air.ir import *
from air.dialects import func
from air.dialects import linalg

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
    print(CostModel().op_stats(module.operation))