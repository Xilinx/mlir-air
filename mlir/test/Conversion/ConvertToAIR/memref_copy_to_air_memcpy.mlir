//===- memref_copy_to_air_memcpy.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma

// Ranked memref.
// CHECK: func.func @func0
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1{{.*}}])
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%c0{{.*}}, %{{.*}}] [%c64{{.*}}, %c16{{.*}}] [%c64{{.*}}, %c1{{.*}}])
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %{{.*}}] [%c16{{.*}}, %c16{{.*}}] [%c64{{.*}}, %c1{{.*}}])
// CHECK: air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}] [%c16{{.*}}, %c16{{.*}}] [%c64{{.*}}, %c1{{.*}}], %{{.*}}[] [] [])
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
#map0 = affine_map<(d0, d1, d2)[s0] -> (d0 * 524288 + s0 + d1 * 512 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module  {
  func.func @func0(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<64x64xf32>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c16, %c16) {
      %1 = memref.subview %arg0[%arg2, 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #map>
      %2 = memref.subview %arg1[0, %arg3] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #map>
      %3 = memref.subview %0[%arg2, %arg3] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #map>
      %4 = memref.alloc() : memref<16x64xf32, 2>
      %5 = memref.alloc() : memref<64x16xf32, 2>
      %6 = memref.alloc() : memref<16x16xf32, 2>
      memref.copy %1, %4 : memref<16x64xf32, #map> to memref<16x64xf32, 2>
      memref.copy %2, %5 : memref<64x16xf32, #map> to memref<64x16xf32, 2>
      memref.copy %3, %6 : memref<16x16xf32, #map> to memref<16x16xf32, 2>
      linalg.matmul ins(%4, %5 : memref<16x64xf32, 2>, memref<64x16xf32, 2>) outs(%6 : memref<16x16xf32, 2>)
      memref.copy %6, %3 : memref<16x16xf32, 2> to memref<16x16xf32, #map>
      memref.dealloc %4 : memref<16x64xf32, 2>
      memref.dealloc %5 : memref<64x16xf32, 2>
      memref.dealloc %6 : memref<16x16xf32, 2>
    }
    return %0 : memref<64x64xf32>
  }
  
// Unranked memref.
// CHECK: func.func @func1
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%c0{{.*}}, %{{.*}}] [%c32{{.*}}, %c32{{.*}}] [%c64{{.*}}, %c1{{.*}}])
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%c0{{.*}}, %{{.*}}] [%c32{{.*}}, %c32{{.*}}] [%c64{{.*}}, %c1{{.*}}])
  func.func @func1(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.muli %arg6, %c32_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %arg7, %c32_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %1, %c64 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [32, 32], strides: [%c64, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<32x32xf32, 2>
    memref.copy %reinterpret_cast, %alloc : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<32x32xf32, 2>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32, 2> to tensor<32x32xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [32, 32], strides: [%c64, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<32x32xf32, 2>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<32x32xf32, 2>
    %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xf32, 2> to tensor<32x32xf32>
    %7 = tensor.empty() : tensor<32x32xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = linalg.matmul ins(%5, %6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return
  }
}
