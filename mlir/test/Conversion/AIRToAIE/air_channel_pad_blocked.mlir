//===- air_channel_pad_blocked.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" | FileCheck %s

// Two cores read 256-row K steps of a K x N tile in blocks of 8x8. The second
// step has 192 rows, padded to 256 by 8 more blocks. Its K block axis and
// in-block K axis are contiguous, but merging them would leave the padding
// counted in blocks on an axis counted in rows, so the padded walk keeps its
// four dimensions. The unpadded walk is merged as before.

// CHECK:       aie.memtile_dma
// CHECK-DAG:     aie.dma_bd({{.*}} : memref<960x32xbf16, 1> offset = 0 len = 8192 sizes = [4, 256, 8] strides = [8, 32, 1])
// CHECK-DAG:     aie.dma_bd({{.*}} : memref<960x32xbf16, 1> offset = 8192 len = 8192
// CHECK-SAME:        sizes = [4, 24, 8, 8] strides = [8, 256, 32, 1]
// CHECK-SAME:        pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 8>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]

module {
  air.channel @L3ToL2 [1, 1]
  air.channel @L2ToL1 [1, 2]
  func.func @pad_blocked(%arg0: memref<960x32xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<960x32xbf16> {
      air.channel.put @L3ToL2[] (%a[] [] []) : (memref<960x32xbf16>)
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %alloc_l2 = memref.alloc() : memref<960x32xbf16, 1>
        air.channel.get @L3ToL2[] (%alloc_l2[] [] []) : (memref<960x32xbf16, 1>)
        air.channel.put @L2ToL1[%c0_s, %c0_s] (%alloc_l2[0, 0, 0, 0, 0, 0] [1, 1, 4, 32, 8, 8] [8192, 8192, 8, 256, 32, 1])
            : (memref<960x32xbf16, 1>)
        air.channel.put @L2ToL1[%c0_s, %c1_s] (%alloc_l2[0, 0, 0, 0, 256, 0] [1, 1, 4, 24, 8, 8] [6144, 6144, 8, 256, 32, 1])
            {pad_before = array<i32: 0, 0, 0, 0, 0, 0>, pad_after = array<i32: 0, 0, 0, 8, 0, 0>}
            : (memref<960x32xbf16, 1>)
        air.herd @herd_0 tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c2_s) {
          %alloc_l1 = memref.alloc() : memref<1x1x4x32x8x8xbf16, 2>
          air.channel.get @L2ToL1[%hx, %hy] (%alloc_l1[] [] []) : (memref<1x1x4x32x8x8xbf16, 2>)
          memref.dealloc %alloc_l1 : memref<1x1x4x32x8x8xbf16, 2>
        }
        memref.dealloc %alloc_l2 : memref<960x32xbf16, 1>
      }
    }
    return
  }
}
