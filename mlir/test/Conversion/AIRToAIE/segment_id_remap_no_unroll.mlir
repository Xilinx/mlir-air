//===- segment_id_remap_no_unroll.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that segment iteration IDs are remapped to constants even when
// totalUnroll == 1 (segment sizes = [1, 1]).
//
// When a segment has sizes [1, 1], there is only one unroll iteration, so
// no segment_unroll_x/y attributes are set on the aie.device. However,
// the segment IDs (block arguments) may still be used as channel indices.
// These must be remapped to constant 0 so that specializeChannelBundle can
// resolve the channel bundle positions.
//
// Without the fix, seg_x/seg_y remain as variables, and channel operations
// using them as indices fail to specialize, causing:
//   'air.channel.get' op failed to get MM2S tile for L3 allocation.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu1_1col' | FileCheck %s

// CHECK: aie.device
// Verify shim DMA allocations exist (would fail without the fix)
// CHECK: aie.shim_dma_allocation @air_channel_3
// CHECK: aie.shim_dma_allocation @air_channel_0

module {
  // L3 <-> L2 channels: indexed by segment unroll IDs [1, 1]
  air.channel @channel_0 [1, 1]
  air.channel @channel_3 [1, 1]
  // L2 <-> L1 channels: internal to segment
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  func.func @test_segment_id_remap(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<64xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // L3 -> L2 put at launch level
      air.channel.put @channel_0[%c0, %c0] (%in[] [] []) {id = 1 : i32} : (memref<64xbf16>)
      // Segment with sizes [1, 1]: totalUnroll == 1.
      // The segment IDs %seg_x, %seg_y are used as channel indices below.
      %1 = air.segment @segment0 async unroll(%seg_x, %seg_y) in (%sx=%c1, %sy=%c1)
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_seg = arith.constant 1 : index
        %l2_buf = memref.alloc() : memref<64xbf16, 1>
        %l2_out = memref.alloc() : memref<64xbf16, 1>
        // L3 -> L2: uses segment IDs as channel indices
        air.channel.get @channel_0[%seg_x, %seg_y] (%l2_buf[] [] []) {id = 2 : i32} : (memref<64xbf16, 1>)
        // L2 -> L1
        air.channel.put @channel_1[] (%l2_buf[] [] []) {id = 3 : i32} : (memref<64xbf16, 1>)
        air.herd @herd0 tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg)
            attributes {id = 3 : i32} {
          %l1_in = memref.alloc() : memref<64xbf16, 2>
          %l1_out = memref.alloc() : memref<64xbf16, 2>
          air.channel.get @channel_1[%tx, %ty] (%l1_in[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)
          air.channel.put @channel_2[%tx, %ty] (%l1_out[] [] []) {id = 5 : i32} : (memref<64xbf16, 2>)
          memref.dealloc %l1_in : memref<64xbf16, 2>
          memref.dealloc %l1_out : memref<64xbf16, 2>
        }
        // L1 -> L2
        air.channel.get @channel_2[] (%l2_out[] [] []) {id = 6 : i32} : (memref<64xbf16, 1>)
        // L2 -> L3: uses segment IDs as channel indices
        air.channel.put @channel_3[%seg_x, %seg_y] (%l2_out[] [] []) {id = 7 : i32} : (memref<64xbf16, 1>)
        memref.dealloc %l2_buf : memref<64xbf16, 1>
        memref.dealloc %l2_out : memref<64xbf16, 1>
      }
      // L2 -> L3 get at launch level
      air.channel.get @channel_3[%c0, %c0] (%out[] [] []) {id = 8 : i32} : (memref<64xbf16>)
    }
    return
  }
}
