// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s

// The scf.parallel should return an asynchronous token which memref.copy depends on
// It should also generate an scf.reduce at the end of its body
// CHECK: %[[EVENT0:.*]] = scf.parallel
// CHECK: scf.reduce
// CHECK: %[[EVENT1:.*]] = air.region async [{{.*}}%[[EVENT0]]{{.*}}]

#map = affine_map<()[s0] -> (s0 * 32)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<384x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<384x1024xf32>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c384 = arith.constant 384 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<384x1024xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<384x1024xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<384x1024xf32>)
    memref.copy %0, %1 : memref<384x1024xf32> to memref<384x1024xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c384, %c1024) step (%c64, %c64) {
      scf.for %arg5 = %c0 to %c1024 step %c64 {
        %2 = memref.alloc() : memref<64x64xf32, 1>
        %3 = memref.alloc() : memref<64x64xf32, 1>
        %4 = memref.alloc() : memref<64x64xf32, 1>
        air.dma_memcpy_nd (%2[] [] [], %arg0[%arg3, %arg5] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xf32, 1>, memref<384x1024xf32>)
        air.dma_memcpy_nd (%3[] [] [], %arg1[%arg5, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xf32, 1>, memref<1024x1024xf32>)
        air.dma_memcpy_nd (%4[] [] [], %1[%arg3, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xf32, 1>, memref<384x1024xf32>)
        air.herd  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%2, %arg11=%3, %arg12=%4) : memref<64x64xf32, 1>, memref<64x64xf32, 1>, memref<64x64xf32, 1> attributes {sym_name = "herd_0"} {
          %c1_0 = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %c64_1 = arith.constant 64 : index
          %c0_2 = arith.constant 0 : index
          %5 = affine.apply #map()[%arg6]
          %6 = affine.apply #map()[%arg7]
          scf.for %arg13 = %c0_2 to %c64_1 step %c32 {
            %7 = memref.alloc() : memref<32x32xf32, 2>
            %8 = memref.alloc() : memref<32x32xf32, 2>
            %9 = memref.alloc() : memref<32x32xf32, 2>
            air.dma_memcpy_nd (%7[] [] [], %arg10[%5, %arg13] [%c32, %c32] [%c64_1, %c1_0]) {id = 4 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32, 1>)
            air.dma_memcpy_nd (%8[] [] [], %arg11[%arg13, %6] [%c32, %c32] [%c64_1, %c1_0]) {id = 5 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32, 1>)
            air.dma_memcpy_nd (%9[] [] [], %arg12[%5, %6] [%c32, %c32] [%c64_1, %c1_0]) {id = 6 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32, 1>)
            linalg.matmul ins(%7, %8 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%9 : memref<32x32xf32, 2>)
            air.dma_memcpy_nd (%arg12[%5, %6] [%c32, %c32] [%c64_1, %c1_0], %9[] [] []) {id = 7 : i32} : (memref<64x64xf32, 1>, memref<32x32xf32, 2>)
            memref.dealloc %7 : memref<32x32xf32, 2>
            memref.dealloc %8 : memref<32x32xf32, 2>
            memref.dealloc %9 : memref<32x32xf32, 2>
          }
          air.herd_terminator
        }
        air.dma_memcpy_nd (%1[%arg3, %arg4] [%c64, %c64] [%c1024, %c1], %4[] [] []) {id = 8 : i32} : (memref<384x1024xf32>, memref<64x64xf32, 1>)
        memref.dealloc %2 : memref<64x64xf32, 1>
        memref.dealloc %3 : memref<64x64xf32, 1>
        memref.dealloc %4 : memref<64x64xf32, 1>
      }
      scf.yield
    }
    memref.copy %1, %arg2 : memref<384x1024xf32> to memref<384x1024xf32>
    return
  }
}

