// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

// RUN: air-opt %s -air-dependency | FileCheck %s

// The second herd should depend on the first
// CHECK: %[[EVENT0:.*]] = air.herd @herd_0 async
// CHECK: %[[EVENT1:.*]] = air.herd @herd_1 async [{{.*}}%[[EVENT0]]{{.*}}]
#map0 = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "MMult_Mult"} {
  func.func @forward(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<128x128xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
    linalg.fill ins(%cst : f32) outs(%1 : memref<128x128xf32>)
    memref.copy %1, %2 : memref<128x128xf32> to memref<128x128xf32>
    air.herd  tile (%arg4, %arg5) in (%arg6=%c8, %arg7=%c2) args(%arg8=%arg1, %arg9=%arg2, %arg10=%2) : memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> attributes {sym_name = "herd_0"} {
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %3 = affine.apply #map0()[%arg4]
      %4 = affine.apply #map1()[%arg5]
      scf.for %arg11 = %c0 to %c128 step %c32 {
        %5 = memref.alloc() : memref<16x32xf32, 2>
        %6 = memref.alloc() : memref<32x64xf32, 2>
        %7 = memref.alloc() : memref<16x64xf32, 2>
        air.dma_memcpy_nd (%5[] [] [], %arg8[%3, %arg11] [%c16, %c32] [%c128, %c1_0]) {id = 1 : i32} : (memref<16x32xf32, 2>, memref<128x128xf32>)
        air.dma_memcpy_nd (%6[] [] [], %arg9[%arg11, %4] [%c32, %c64] [%c128, %c1_0]) {id = 2 : i32} : (memref<32x64xf32, 2>, memref<128x128xf32>)
        air.dma_memcpy_nd (%7[] [] [], %arg10[%3, %4] [%c16, %c64] [%c128, %c1_0]) {id = 3 : i32} : (memref<16x64xf32, 2>, memref<128x128xf32>)
        linalg.matmul ins(%5, %6 : memref<16x32xf32, 2>, memref<32x64xf32, 2>) outs(%7 : memref<16x64xf32, 2>)
        air.dma_memcpy_nd (%arg10[%3, %4] [%c16, %c64] [%c128, %c1_0], %7[] [] []) {id = 4 : i32} : (memref<128x128xf32>, memref<16x64xf32, 2>)
        memref.dealloc %5 : memref<16x32xf32, 2>
        memref.dealloc %6 : memref<32x64xf32, 2>
        memref.dealloc %7 : memref<16x64xf32, 2>
      }
      air.herd_terminator
    }
    air.herd  tile (%arg4, %arg5) in (%arg6=%c8, %arg7=%c1) args(%arg8=%arg0, %arg9=%2, %arg10=%0) : memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> attributes {sym_name = "herd_1"} {
      %c1_0 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %c0 = arith.constant 0 : index
      %3 = affine.apply #map0()[%arg4]
      %4 = memref.alloc() : memref<16x128xf32, 2>
      %5 = memref.alloc() : memref<16x128xf32, 2>
      %6 = memref.alloc() : memref<16x128xf32, 2>
      air.dma_memcpy_nd (%4[] [] [], %arg8[%3, %c0] [%c16, %c128] [%c128, %c1_0]) {id = 5 : i32} : (memref<16x128xf32, 2>, memref<128x128xf32>)
      air.dma_memcpy_nd (%5[] [] [], %arg9[%3, %c0] [%c16, %c128] [%c128, %c1_0]) {id = 6 : i32} : (memref<16x128xf32, 2>, memref<128x128xf32>)
      air.dma_memcpy_nd (%6[] [] [], %arg10[%3, %c0] [%c16, %c128] [%c128, %c1_0]) {id = 7 : i32} : (memref<16x128xf32, 2>, memref<128x128xf32>)
      linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<16x128xf32, 2>, memref<16x128xf32, 2>) outs(%6 : memref<16x128xf32, 2>) {
      ^bb0(%arg11: f32, %arg12: f32, %arg13: f32):
        %7 = arith.mulf %arg11, %arg12 : f32
        linalg.yield %7 : f32
      }
      air.dma_memcpy_nd (%arg10[%3, %c0] [%c16, %c128] [%c128, %c1_0], %6[] [] []) {id = 8 : i32} : (memref<128x128xf32>, memref<16x128xf32, 2>)
      memref.dealloc %4 : memref<16x128xf32, 2>
      memref.dealloc %5 : memref<16x128xf32, 2>
      memref.dealloc %6 : memref<16x128xf32, 2>
      air.herd_terminator
    }
    memref.copy %0, %arg3 : memref<128x128xf32> to memref<128x128xf32>
    return
  }
}