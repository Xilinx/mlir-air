// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

// RUN: air-opt %s -air-dependency | FileCheck %s

// The scf.parallel should return an asynchronous token which memref.copy depends on
// It should also generate an scf.reduce at the end of its body
// CHECK: %[[EVENT0:.*]] = scf.parallel
// CHECK: scf.reduce
// CHECK: %[[EVENT1:.*]] = air.execute async [{{.*}}%[[EVENT0]]{{.*}}]

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<1024x1024xf32>)
    memref.copy %0, %1 : memref<1024x1024xf32> to memref<1024x1024xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c1024, %c1024) step (%c128, %c128) {
      scf.for %arg5 = %c0 to %c1024 step %c32 {
        air.herd  tile (%arg6, %arg7) in (%arg8=%c4, %arg9=%c4) args(%arg10=%arg3, %arg11=%arg5, %arg12=%arg0, %arg13=%arg4, %arg14=%arg1, %arg15=%1) : index, index, memref<1024x1024xf32>, index, memref<1024x1024xf32>, memref<1024x1024xf32> attributes {sym_name = "herd_0"} {
          %c1 = arith.constant 1 : index
          %c1024_0 = arith.constant 1024 : index
          %c32_1 = arith.constant 32 : index
          %2 = affine.apply #map()[%arg6]
          %3 = affine.apply #map()[%arg7]
          %4 = arith.addi %arg10, %2 : index
          %5 = arith.addi %arg13, %3 : index
          %6 = memref.alloc() : memref<32x32xf32, 2>
          %7 = memref.alloc() : memref<32x32xf32, 2>
          %8 = memref.alloc() : memref<32x32xf32, 2>
          air.dma_memcpy_nd (%6[] [] [], %arg12[%4, %arg11] [%c32_1, %c32_1] [%c1024_0, %c1]) {id = 1 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
          air.dma_memcpy_nd (%7[] [] [], %arg14[%arg11, %5] [%c32_1, %c32_1] [%c1024_0, %c1]) {id = 2 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
          air.dma_memcpy_nd (%8[] [] [], %arg15[%4, %5] [%c32_1, %c32_1] [%c1024_0, %c1]) {id = 3 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
          linalg.matmul ins(%6, %7 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%8 : memref<32x32xf32, 2>)
          air.dma_memcpy_nd (%arg15[%4, %5] [%c32_1, %c32_1] [%c1024_0, %c1], %8[] [] []) {id = 4 : i32} : (memref<1024x1024xf32>, memref<32x32xf32, 2>)
          memref.dealloc %6 : memref<32x32xf32, 2>
          memref.dealloc %7 : memref<32x32xf32, 2>
          memref.dealloc %8 : memref<32x32xf32, 2>
          air.herd_terminator
        }
      }
      scf.yield
    }
    memref.copy %1, %arg2 : memref<1024x1024xf32> to memref<1024x1024xf32>
    return
  }
}

