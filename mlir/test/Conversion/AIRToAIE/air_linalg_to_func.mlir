//===- air_linalg_to_func.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===--------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-to-func | FileCheck %s

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<()[s0] -> (s0 * 4)>
module {
  // CHECK: func.func private @linalg_fill_f32_view64x64xf32as2(f32, memref<64x64xf32, 2>) 
  // CHECK: func.func private @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(memref<64x64xbf16, 2>, memref<16x4x64xbf16, 2>, memref<64x64xf32, 2>) 
  func.func @forward(%arg0: memref<512x128xbf16>, %arg1: memref<32x4x4x128xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.forall (%arg3, %arg4) in (4, 4) {
      %0 = affine.apply #map(%arg3)
      %1 = affine.apply #map(%arg4)
      %alloc = memref.alloc() : memref<128x128xbf16, 1>
      air.dma_memcpy_nd (%alloc[] [] [], %arg0[%0, %c0] [%c128, %c128] [%c128, %c1]) {id = 1 : i32} : (memref<128x128xbf16, 1>, memref<512x128xbf16>)
      %alloc_0 = memref.alloc() : memref<32x4x128xbf16, 1>
      air.dma_memcpy_nd (%alloc_0[] [] [], %arg1[%c0, %arg4, %c0, %c0] [%c32, %c1, %c4, %c128] [%c2048, %c512, %c128, %c1]) {id = 2 : i32} : (memref<32x4x128xbf16, 1>, memref<32x4x4x128xbf16>)
      %alloc_1 = memref.alloc() : memref<128x128xf32, 1>
      scf.forall (%arg5, %arg6) in (2, 2) {
        %2 = affine.apply #map1(%arg5)
        %3 = affine.apply #map1(%arg6)
        %alloc_2 = memref.alloc() : memref<64x64xf32, 2>
        linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<64x64xf32, 2>)
        // CHECK: linalg_fill_f32_view64x64xf32as2
        scf.for %arg7 = %c0 to %c32 step %c16 {
          %4 = affine.apply #map2()[%arg7]
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 2>
          %alloc_4 = memref.alloc() : memref<16x4x64xbf16, 2>
          air.dma_memcpy_nd (%alloc_3[] [] [], %alloc[%2, %4] [%c64, %c64] [%c128, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 2>, memref<128x128xbf16, 1>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %alloc_0[%arg7, %c0, %3] [%c16, %c4, %c64] [%c512, %c128, %c1]) {id = 4 : i32} : (memref<16x4x64xbf16, 2>, memref<32x4x128xbf16, 1>)
          %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1], [2]] : memref<16x4x64xbf16, 2> into memref<64x64xbf16, 2>
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_3, %collapse_shape : memref<64x64xbf16, 2>, memref<64x64xbf16, 2>) outs(%alloc_2 : memref<64x64xf32, 2>)
          // CHECK: linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(%alloc_3, %alloc_4, %alloc_2) : (memref<64x64xbf16, 2>, memref<16x4x64xbf16, 2>, memref<64x64xf32, 2>) -> ()
          // CHECK-NOT: linalg.matmul
          memref.dealloc %alloc_3 : memref<64x64xbf16, 2>
          memref.dealloc %alloc_4 : memref<16x4x64xbf16, 2>
        }
        air.dma_memcpy_nd (%alloc_1[%2, %3] [%c64, %c64] [%c128, %c1], %alloc_2[] [] []) {id = 5 : i32} : (memref<128x128xf32, 1>, memref<64x64xf32, 2>)
        memref.dealloc %alloc_2 : memref<64x64xf32, 2>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      air.dma_memcpy_nd (%arg2[%0, %1] [%c128, %c128] [%c512, %c1], %alloc_1[] [] []) {id = 6 : i32} : (memref<512x512xf32>, memref<128x128xf32, 1>)
      memref.dealloc %alloc : memref<128x128xbf16, 1>
      memref.dealloc %alloc_0 : memref<32x4x128xbf16, 1>
      memref.dealloc %alloc_1 : memref<128x128xf32, 1>
    }
    return %arg2 : memref<512x512xf32>
  }
}
