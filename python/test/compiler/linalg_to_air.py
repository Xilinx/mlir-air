# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
from air.mlir.ir import *
from air.mlir.passmanager import PassManager
from air.mlir.dialects import func
from air.mlir.dialects import linalg
from air.mlir.dialects import arith
from air.mlir.dialects import tensor

# this has a side effect of registering the air passes
import air.compiler.util

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f

# CHECK-LABEL: TEST: matmul_l1_l2_2x2
# CHECK:scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c128, %c128) step (%c64, %c64) {
# CHECK: scf.for %arg4 = %c0 to %c128 step %c64 {
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<128x128xi32>
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<128x128xi32>
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<128x128xi32>
# CHECK: air.launch_herd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c2)
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>
# CHECK: air.herd_terminator
# CHECK: air.dma_memcpy_nd ({{.*}}) {id = 8 : i32} : (memref<128x128xi32>, memref<64x64xi32, 1>
@run
def matmul_l1_l2_2x2():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      elemTy = IntegerType.get_signless(32)
      @func.FuncOp.from_py_func(
        RankedTensorType.get((128, 128), elemTy), RankedTensorType.get((128, 128), elemTy))
      def matmul_on_tensors(lhs, rhs):
        zero = arith.ConstantOp(elemTy, IntegerAttr.get(elemTy, 0))
        init_tensor = linalg.InitTensorOp((128, 128), elemTy)
        zero_tensor = linalg.fill(zero.result, outs=[init_tensor.result])
        out = linalg.matmul(lhs, rhs, outs=[zero_tensor])
        return out
    PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE).run(module)
    PassManager.parse('air-linalg-codegen{l1-tile-size=32,32,32 l2-tile-size=64,64,64},air-par-to-herd{depth=1},air-copy-to-dma').run(module)
    print(module)
