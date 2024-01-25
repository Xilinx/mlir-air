//===- condense_memref_ops_to_air_memcpy.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma -canonicalize -cse | FileCheck %s

// CHECK:  %[[CST128:.*]] = arith.constant 128 : index
// CHECK:  %[[CST32:.*]] = arith.constant 32 : index
// CHECK:  %[[CST8:.*]] = arith.constant 8 : index
// CHECK:  %[[CST16:.*]] = arith.constant 16 : index
// CHECK:  %[[CST0:.*]] = arith.constant 0 : index
// CHECK:  %[[CST1:.*]] = arith.constant 1 : index
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %[[CST0]]] [%[[CST8]], %[[CST16]]] [%[[CST16]], %[[CST1]]]) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[CST0]], %{{.*}}] [%[[CST16]], %[[CST16]]] [%[[CST32]], %[[CST1]]]) : (memref<1x1x16x16xi32, 1>, memref<16x32xi32>)
// CHECK:  air.herd @herd_0
// CHECK:  %[[CST32_0:.*]] = arith.constant 32 : index
// CHECK:  %[[CST4_0:.*]] = arith.constant 4 : index
// CHECK:  %[[CST2_0:.*]] = arith.constant 2 : index
// CHECK:  %[[CST1_0:.*]] = arith.constant 1 : index
// CHECK:  %[[CST16_0:.*]] = arith.constant 16 : index
// CHECK:  %[[CST64_0:.*]] = arith.constant 64 : index
// CHECK:  %[[CST8_0:.*]] = arith.constant 8 : index
// CHECK:  %[[CST128_0:.*]] = arith.constant 128 : index
// CHECK:  %[[CST0_0:.*]] = arith.constant 0 : index
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %[[CST0_0]], %[[CST0_0]], %[[CST0_0]]] [%[[CST2_0]], %[[CST2_0]], %[[CST4_0]], %[[CST8_0]]] [%[[CST8_0]], %[[CST64_0]], %[[CST16_0]], %[[CST1_0]]]) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[CST0_0]], %{{.*}}, %[[CST0_0]], %[[CST0_0]]] [%[[CST2_0]], %[[CST2_0]], %[[CST8_0]], %[[CST8_0]]] [%[[CST8_0]], %[[CST128_0]], %[[CST16_0]], %[[CST1_0]]]) : (memref<1x1x2x2x8x8xi32, 2>, memref<1x1x16x16xi32, 1>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}, %[[CST0_0]], %[[CST0_0]]] [%[[CST1_0]], %[[CST1_0]], %[[CST8_0]], %[[CST16_0]]] [%[[CST128_0]], %[[CST128_0]], %[[CST16_0]], %[[CST1_0]]], %{{.*}}[%[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]]] [%[[CST1_0]], %[[CST1_0]], %[[CST2_0]], %[[CST4_0]], %[[CST2_0]], %[[CST8_0]]] [%[[CST128_0]], %[[CST128_0]], %[[CST32_0]], %[[CST8_0]], %[[CST64_0]], %[[CST1_0]]]) : (memref<1x1x8x16xi32, 1>, memref<1x1x2x2x4x8xi32, 2>)
// CHECK:  air.herd_terminator
// CHECK:  air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}] [%[[CST8]], %[[CST16]]] [%[[CST32]], %[[CST1]]], %{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST8]], %[[CST16]]] [%[[CST128]], %[[CST128]], %[[CST16]], %[[CST1]]]) : (memref<8x32xi32>, memref<1x1x8x16xi32, 1>)

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
module {
  func.func @func0(%0 : memref<8x16xi32>, %1 : memref<16x32xi32>, %2 : memref<8x32xi32>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    air.launch (%arg0, %arg1) in (%arg2=%c1, %arg3=%c2) args(%arg4=%0, %arg5=%1, %arg6=%2) : memref<8x16xi32>, memref<16x32xi32>, memref<8x32xi32> {
      air.segment @segment_0  args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg4, %arg10=%arg5, %arg11=%arg6) : index, index, memref<8x16xi32>, memref<16x32xi32>, memref<8x32xi32> {
        %c1_0 = arith.constant 1 : index
        %3 = affine.apply #map()[%arg7]
        %4 = affine.apply #map1()[%arg8]
        %subview = memref.subview %arg9[%3, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
        %subview_1 = memref.subview %arg10[0, %4] [16, 16] [1, 1] : memref<16x32xi32> to memref<16x16xi32, strided<[32, 1], offset: ?>>
        %subview_2 = memref.subview %arg11[%3, %4] [8, 16] [1, 1] : memref<8x32xi32> to memref<8x16xi32, strided<[32, 1], offset: ?>>
        %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
        %transpose = memref.transpose %subview (d0, d1) -> (d0, d1) : memref<8x16xi32, strided<[16, 1], offset: ?>> to memref<8x16xi32, strided<[16, 1], offset: ?>>
        air.dma_memcpy_nd (%alloc[] [] [], %transpose[] [] []) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32, strided<[16, 1], offset: ?>>)
        %alloc_3 = memref.alloc() : memref<1x1x16x16xi32, 1>
        %transpose_4 = memref.transpose %subview_1 (d0, d1) -> (d0, d1) : memref<16x16xi32, strided<[32, 1], offset: ?>> to memref<16x16xi32, strided<[32, 1], offset: ?>>
        air.dma_memcpy_nd (%alloc_3[] [] [], %transpose_4[] [] []) : (memref<1x1x16x16xi32, 1>, memref<16x16xi32, strided<[32, 1], offset: ?>>)
        %alloc_5 = memref.alloc() : memref<1x1x8x16xi32, 1>
        air.herd @herd_0  tile (%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c1_0) args(%arg16=%alloc, %arg17=%alloc_3, %arg18=%alloc_5) : memref<1x1x8x16xi32, 1>, memref<1x1x16x16xi32, 1>, memref<1x1x8x16xi32, 1> {
          %c0_i32 = arith.constant 0 : i32
          %subview_8 = memref.subview %arg16[%arg12, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
          %subview_9 = memref.subview %arg17[0, %arg13, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xi32, 1> to memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1>
          %subview_10 = memref.subview %arg18[%arg12, %arg13, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
          %alloc_11 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
          %expand_shape = memref.expand_shape %subview_8 [[0], [1], [2, 3], [4, 5]] : memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1> into memref<1x1x2x4x2x8xi32, strided<[128, 128, 64, 16, 8, 1], offset: ?>, 1>
          %transpose_12 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x4x2x8xi32, strided<[128, 128, 64, 16, 8, 1], offset: ?>, 1> to memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1], offset: ?>, 1>
          air.dma_memcpy_nd (%alloc_11[] [] [], %transpose_12[] [] []) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1], offset: ?>, 1>)
          %alloc_13 = memref.alloc() : memref<1x1x2x2x8x8xi32, 2>
          %expand_shape_14 = memref.expand_shape %subview_9 [[0], [1], [2, 3], [4, 5]] : memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1> into memref<1x1x2x8x2x8xi32, strided<[256, 256, 128, 16, 8, 1], offset: ?>, 1>
          %transpose_15 = memref.transpose %expand_shape_14 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x8x2x8xi32, strided<[256, 256, 128, 16, 8, 1], offset: ?>, 1> to memref<1x1x2x2x8x8xi32, strided<[256, 256, 8, 128, 16, 1], offset: ?>, 1>
          air.dma_memcpy_nd (%alloc_13[] [] [], %transpose_15[] [] []) : (memref<1x1x2x2x8x8xi32, 2>, memref<1x1x2x2x8x8xi32, strided<[256, 256, 8, 128, 16, 1], offset: ?>, 1>)
          %alloc_16 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
          %transpose_17 = memref.transpose %alloc_16 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x2x2x4x8xi32, 2> to memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>
          air.dma_memcpy_nd (%subview_10[] [] [], %transpose_17[] [] []) : (memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>, memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>)
          memref.dealloc %alloc_11 : memref<1x1x2x2x4x8xi32, 2>
          memref.dealloc %alloc_13 : memref<1x1x2x2x8x8xi32, 2>
          memref.dealloc %alloc_16 : memref<1x1x2x2x4x8xi32, 2>
          air.herd_terminator
        }
        %subview_6 = memref.subview %alloc_5[0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<8x16xi32, 1>
        %transpose_7 = memref.transpose %subview_6 (d0, d1) -> (d0, d1) : memref<8x16xi32, 1> to memref<8x16xi32, strided<[16, 1]>, 1>
        air.dma_memcpy_nd (%subview_2[] [] [], %transpose_7[] [] []) : (memref<8x16xi32, strided<[32, 1], offset: ?>>, memref<8x16xi32, strided<[16, 1]>, 1>)
        memref.dealloc %alloc_3 : memref<1x1x16x16xi32, 1>
        memref.dealloc %alloc : memref<1x1x8x16xi32, 1>
        memref.dealloc %alloc_5 : memref<1x1x8x16xi32, 1>
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
