//===- segment_unroll_metadata_ordering.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that metadataArray entries are correctly reordered for segment unroll
// with non-square channel dimensions. The output channel has size=[2]
// (num_heads_per_unroll), and segment unroll creates 2 devices with 2 tiles
// each, giving 4 shim allocations. The metadataArray must be linearized to
// match getIteratorFromMDVector's expected order.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// The output channel @out_chan has size=[2] (one per segment unroll copy).
// After segment unroll, shim allocations are created for each (tile, unroll).
// The metadataArray on the launch-body ChannelGet must be ordered so that
// getIteratorFromMDVector(channelDims, [tileIdx, unrollCopy]) maps to the
// correct shim allocation.

// Check that both devices are created with correct allocations:
// CHECK-LABEL: aie.device{{.*}}@segment_meta_0_0
// CHECK:       aie.shim_dma_allocation @air_out_chan_0_0
// CHECK:       segment_unroll_x = 0

// CHECK-LABEL: aie.device{{.*}}@segment_meta_1_0
// CHECK:       aie.shim_dma_allocation @air_out_chan_1_0
// CHECK:       segment_unroll_x = 1

// Check metadataArray ordering on the launch-body channel gets.
// With segment unroll, entries should include allocations from both devices.
// The metadataArray must be ordered to match getIteratorFromMDVector.
// CHECK: air.channel.get @out_chan[%c0]
// CHECK-SAME: metadataArray = [{base = "air_out_chan_0_0"
// CHECK-SAME:                   {base = "air_out_chan_1_0"
// CHECK: air.channel.get @out_chan[%c1]
// CHECK-SAME: metadataArray = [{base = "air_out_chan_0_0"
// CHECK-SAME:                   {base = "air_out_chan_1_0"

module {
  air.channel @out_chan [2]

  func.func @test_metadata_ordering(%arg0: memref<128xbf16>, %out: memref<128xbf16>) {
    %0 = air.launch async () in () args(%input=%arg0, %output=%out) : memref<128xbf16>, memref<128xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index

      // Output channel get at launch level (L2→L3)
      air.channel.get @out_chan[%c0] (%output[%c0] [%c64] [%c1]) {id = 10 : i32} : (memref<128xbf16>)
      air.channel.get @out_chan[%c1] (%output[%c64] [%c64] [%c1]) {id = 11 : i32} : (memref<128xbf16>)

      // 2x1 segment unroll creates two isolated devices
      %segment = air.segment @segment_meta async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1)
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 2 : i64} {
        %c0_seg = arith.constant 0 : index
        %c1_seg = arith.constant 1 : index

        // L2 output buffer
        %l2_out = memref.alloc() : memref<64xbf16, 1>

        // Output channel put from segment to L3
        air.channel.put @out_chan[%ux] (%l2_out[] [] []) {id = 12 : i32} : (memref<64xbf16, 1>)

        %herd = air.herd @herd_meta async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg)
            attributes {id = 3 : i32} {
          %l1_buf = memref.alloc() : memref<64xbf16, 2>
          memref.dealloc %l1_buf : memref<64xbf16, 2>
        }

        memref.dealloc %l2_out : memref<64xbf16, 1>
      }
    }
    return
  }
}
