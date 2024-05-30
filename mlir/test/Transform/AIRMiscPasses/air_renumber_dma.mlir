//===- air_renumber_dma.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-renumber-dma="mode=herd" | FileCheck %s
// CHECK: id = 1
// CHECK: id = 2
// CHECK: id = 3
// CHECK: id = 4

// CHECK: id = 1
// CHECK: id = 2
// CHECK: id = 3
// CHECK: id = 4

// CHECK: id = 1
// CHECK: id = 2
// CHECK: id = 3
// CHECK: id = 4

// CHECK: id = 1
// CHECK: id = 2
// CHECK: id = 3
// CHECK: id = 4
// CHECK: id = 5

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @forward(%arg0: memref<64x256xf32>, %arg1: memref<256x64xf32>, %arg2: memref<256x64xf32>, %arg3: memref<256x64xf32>) -> memref<64x1xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<64x64xf32>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32>
    memref.copy %0, %1 : memref<64x64xf32> to memref<64x64xf32>
    air.herd @herd_0  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg1, %arg10=%1) : memref<64x256xf32>, memref<256x64xf32>, memref<64x64xf32> {
      %c64 = arith.constant 64 : index
      %c1_1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %19 = affine.apply #map0()[%arg4]
      %20 = affine.apply #map0()[%arg5]
      scf.for %arg11 = %c0 to %c256 step %c32 {
        %21 = memref.alloc() : memref<32x32xf32, 2>
        %22 = memref.alloc() : memref<32x32xf32, 2>
        %23 = memref.alloc() : memref<32x32xf32, 2>
        air.dma_memcpy_nd (%21[] [] [], %arg8[%19, %arg11] [%c32, %c32] [%c256, %c1_1]) {id = 1 : i32} : (memref<32x32xf32, 2>, memref<64x256xf32>)
        air.dma_memcpy_nd (%22[] [] [], %arg9[%arg11, %20] [%c32, %c32] [%c64, %c1_1]) {id = 2 : i32} : (memref<32x32xf32, 2>, memref<256x64xf32>)
        air.dma_memcpy_nd (%23[] [] [], %arg10[%19, %20] [%c32, %c32] [%c64, %c1_1]) {id = 3 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32>)
        linalg.matmul ins(%21, %22 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%23 : memref<32x32xf32, 2>)
        air.dma_memcpy_nd (%arg10[%19, %20] [%c32, %c32] [%c64, %c1_1], %23[] [] []) {id = 4 : i32} : (memref<64x64xf32>, memref<32x32xf32, 2>)
        memref.dealloc %21 : memref<32x32xf32, 2>
        memref.dealloc %22 : memref<32x32xf32, 2>
        memref.dealloc %23 : memref<32x32xf32, 2>
      }
      air.herd_terminator
    }
    %2 = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32>
    linalg.fill ins(%cst : f32) outs(%2 : memref<64x64xf32>)
    %3 = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32>
    memref.copy %2, %3 : memref<64x64xf32> to memref<64x64xf32>
    air.herd @herd_1  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg2, %arg10=%3) : memref<64x256xf32>, memref<256x64xf32>, memref<64x64xf32> {
      %c64 = arith.constant 64 : index
      %c1_1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %19 = affine.apply #map0()[%arg4]
      %20 = affine.apply #map0()[%arg5]
      scf.for %arg11 = %c0 to %c256 step %c32 {
        %21 = memref.alloc() : memref<32x32xf32, 2>
        %22 = memref.alloc() : memref<32x32xf32, 2>
        %23 = memref.alloc() : memref<32x32xf32, 2>
        air.dma_memcpy_nd (%21[] [] [], %arg8[%19, %arg11] [%c32, %c32] [%c256, %c1_1]) {id = 5 : i32} : (memref<32x32xf32, 2>, memref<64x256xf32>)
        air.dma_memcpy_nd (%22[] [] [], %arg9[%arg11, %20] [%c32, %c32] [%c64, %c1_1]) {id = 6 : i32} : (memref<32x32xf32, 2>, memref<256x64xf32>)
        air.dma_memcpy_nd (%23[] [] [], %arg10[%19, %20] [%c32, %c32] [%c64, %c1_1]) {id = 7 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32>)
        linalg.matmul ins(%21, %22 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%23 : memref<32x32xf32, 2>)
        air.dma_memcpy_nd (%arg10[%19, %20] [%c32, %c32] [%c64, %c1_1], %23[] [] []) {id = 8 : i32} : (memref<64x64xf32>, memref<32x32xf32, 2>)
        memref.dealloc %21 : memref<32x32xf32, 2>
        memref.dealloc %22 : memref<32x32xf32, 2>
        memref.dealloc %23 : memref<32x32xf32, 2>
      }
      air.herd_terminator
    }
    %4 = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32>
    linalg.fill ins(%cst : f32) outs(%4 : memref<64x64xf32>)
    %5 = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32>
    memref.copy %4, %5 : memref<64x64xf32> to memref<64x64xf32>
    air.herd @herd_2  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%1, %arg9=%3, %arg10=%5) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
      %c1_1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %19 = affine.apply #map0()[%arg4]
      %20 = affine.apply #map0()[%arg5]
      scf.for %arg11 = %c0 to %c64 step %c32 {
        %21 = memref.alloc() : memref<32x32xf32, 2>
        %22 = memref.alloc() : memref<32x32xf32, 2>
        %23 = memref.alloc() : memref<32x32xf32, 2>
        air.dma_memcpy_nd (%21[] [] [], %arg8[%19, %arg11] [%c32, %c32] [%c64, %c1_1]) {id = 9 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32>)
        air.dma_memcpy_nd (%22[] [] [], %arg9[%arg11, %20] [%c32, %c32] [%c64, %c1_1]) {id = 10 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32>)
        air.dma_memcpy_nd (%23[] [] [], %arg10[%19, %20] [%c32, %c32] [%c64, %c1_1]) {id = 11 : i32} : (memref<32x32xf32, 2>, memref<64x64xf32>)
        linalg.matmul ins(%21, %22 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%23 : memref<32x32xf32, 2>)
        air.dma_memcpy_nd (%arg10[%19, %20] [%c32, %c32] [%c64, %c1_1], %23[] [] []) {id = 12 : i32} : (memref<64x64xf32>, memref<32x32xf32, 2>)
        memref.dealloc %21 : memref<32x32xf32, 2>
        memref.dealloc %22 : memref<32x32xf32, 2>
        memref.dealloc %23 : memref<32x32xf32, 2>
      }
      air.herd_terminator
    }
    %6 = memref.alloc() {alignment = 128 : i64} : memref<64x1xi64>
    linalg.fill ins(%c0_i64 : i64) outs(%6 : memref<64x1xi64>)
    %7 = memref.alloc() {alignment = 128 : i64} : memref<64x1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%7 : memref<64x1xf32>)
    %8 = memref.alloc() {alignment = 128 : i64} : memref<64x1xf32>
    memref.copy %7, %8 : memref<64x1xf32> to memref<64x1xf32>
    %9 = memref.alloc() {alignment = 128 : i64} : memref<64x1xi64>
    memref.copy %6, %9 : memref<64x1xi64> to memref<64x1xi64>
    air.herd @herd_3  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%5, %arg9=%8, %arg10=%9) : memref<64x64xf32>, memref<64x1xf32>, memref<64x1xi64> {
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %19 = affine.apply #map1()[%arg4]
      %20 = memref.alloc() : memref<64x64xf32, 2>
      %21 = memref.alloc() : memref<64x1xf32, 2>
      %22 = memref.alloc() : memref<64x1xi64, 2>
      air.dma_memcpy_nd (%20[] [] [], %arg8[%19, %c0] [%c64, %c64] [%c64, %c1_1]) {id = 13 : i32} : (memref<64x64xf32, 2>, memref<64x64xf32>)
      air.dma_memcpy_nd (%21[] [] [], %arg9[%19, %c0] [%c64, %c1_1] [%c1_1, %c1_1]) {id = 14 : i32} : (memref<64x1xf32, 2>, memref<64x1xf32>)
      air.dma_memcpy_nd (%22[] [] [], %arg10[%19, %c0] [%c64, %c1_1] [%c1_1, %c1_1]) {id = 15 : i32} : (memref<64x1xi64, 2>, memref<64x1xi64>)
      linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%20 : memref<64x64xf32, 2>) outs(%21, %22 : memref<64x1xf32, 2>, memref<64x1xi64, 2>) {
      ^bb0(%arg11: f32, %arg12: f32, %arg13: i64):
        %23 = linalg.index 1 : index
        %24 = arith.index_cast %23 : index to i64
        %25 = arith.cmpf ogt, %arg11, %arg12 : f32
        %26 = arith.select %25, %arg11, %arg12 : f32
        %27 = arith.select %25, %24, %arg13 : i64
        linalg.yield %26, %27 : f32, i64
      }
      air.dma_memcpy_nd (%arg9[%19, %c0] [%c64, %c1_1] [%c1_1, %c1_1], %21[] [] []) {id = 16 : i32} : (memref<64x1xf32>, memref<64x1xf32, 2>)
      air.dma_memcpy_nd (%arg10[%19, %c0] [%c64, %c1_1] [%c1_1, %c1_1], %22[] [] []) {id = 17 : i32} : (memref<64x1xi64>, memref<64x1xi64, 2>)
      memref.dealloc %20 : memref<64x64xf32, 2>
      memref.dealloc %21 : memref<64x1xf32, 2>
      memref.dealloc %22 : memref<64x1xi64, 2>
      air.herd_terminator
    }

    return %8 : memref<64x1xf32>
  }
}
