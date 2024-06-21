//===- time_multiplexing_herds.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds="num-rows=4 num-cols=4 row-anchor=2 col-anchor=0" | FileCheck %s
// CHECK: air.segment {{.*}} attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64}
// CHECK: air.herd {{.*}} attributes {x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd {{.*}} attributes {x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd {{.*}} attributes {x_loc = 0 : i64, y_loc = 2 : i64}

#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @func0(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c4 = arith.constant 4 : index
    air.launch (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) {
      air.segment @segment_0  {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %alloc = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x128x128xbf16, 1 : i32>
        air.herd @herd_0  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%alloc) : memref<1x1x32x32x4x4xbf16, 2 : i32> {
          %cst = arith.constant 0.000000e+00 : bf16
          %0 = affine.apply #map()[%arg7]
          %1 = affine.apply #map()[%arg8]
          %subview = memref.subview %arg11[0, 0, %1, %0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
          linalg.fill ins(%cst : bf16) outs(%subview : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
        }
        scf.for %arg7 = %c1 to %c16 step %c1 {
          air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) args(%arg12=%alloc) : memref<1x1x32x32x4x4xbf16, 2 : i32> {
            %cst = arith.constant 0.000000e+00 : bf16
            %0 = affine.apply #map()[%arg8]
            %1 = affine.apply #map()[%arg9]
            %subview = memref.subview %arg12[0, 0, %1, %0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.fill ins(%cst : bf16) outs(%subview : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>)
          }
        }
        air.herd @herd_0  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%alloc, %arg12=%alloc_0) : memref<1x1x32x32x4x4xbf16, 2 : i32>, memref<1x1x128x128xbf16, 1 : i32> {
          %0 = affine.apply #map()[%arg7]
          %1 = affine.apply #map()[%arg8]
          %subview = memref.subview %arg11[0, 0, %1, %0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
          %transpose = memref.transpose %subview (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32> to memref<1x1x16x4x16x4xbf16, strided<[16384, 16384, 16, 4, 512, 1], offset: ?>, 2 : i32>
          %2 = affine.apply #map1()[%arg7]
          %3 = affine.apply #map1()[%arg8]
          %subview_1 = memref.subview %arg12[0, 0, %2, %3] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x128x128xbf16, 1 : i32> to memref<1x1x64x64xbf16, strided<[16384, 16384, 128, 1], offset: ?>, 1 : i32>
          air.dma_memcpy_nd (%subview_1[] [] [], %transpose[] [] []) : (memref<1x1x64x64xbf16, strided<[16384, 16384, 128, 1], offset: ?>, 1 : i32>, memref<1x1x16x4x16x4xbf16, strided<[16384, 16384, 16, 4, 512, 1], offset: ?>, 2 : i32>)
        }
        memref.dealloc %alloc_0 : memref<1x1x128x128xbf16, 1 : i32>
        memref.dealloc %alloc : memref<1x1x32x32x4x4xbf16, 2 : i32>
      }
    }
    return
  }
}
