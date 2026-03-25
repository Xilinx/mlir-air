//===- shim_dma_sequential_packing.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that multiple channel types (e.g., QKIn and VIn) targeting different
// memtile columns pack sequentially on shim tiles rather than interleaving.
//
// With 4 QKIn channels and 4 VIn channels, the shim allocator should pack
// QKIn stages 2-per-tile on tiles 0-1 (using MM2S:0 and MM2S:1), then
// VIn stages 2-per-tile on tiles 2-3. Previously, the same-column constraint
// caused QKIn and VIn to share tiles (QKIn on MM2S:0, VIn on MM2S:1 of the
// same tile).

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// CHECK: aie.device
// Capture shim tiles for columns 0-3:
// CHECK-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// CHECK-DAG: %[[SHIM1:.*]] = aie.tile(1, 0)
// CHECK-DAG: %[[SHIM2:.*]] = aie.tile(2, 0)
// CHECK-DAG: %[[SHIM3:.*]] = aie.tile(3, 0)
// Verify QKIn channels are packed on tiles 0-1:
// CHECK-DAG: aie.shim_dma_allocation @air_chan_a_0(%[[SHIM0]], MM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_chan_a_1(%[[SHIM0]], MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_chan_a_2(%[[SHIM1]], MM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_chan_a_3(%[[SHIM1]], MM2S, 1)
// Verify VIn channels are on separate tiles 2-3 (NOT sharing with QKIn):
// CHECK-DAG: aie.shim_dma_allocation @air_chan_b_0(%[[SHIM2]], MM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_chan_b_1(%[[SHIM2]], MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_chan_b_2(%[[SHIM3]], MM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_chan_b_3(%[[SHIM3]], MM2S, 1)

module {
  // 4 "QKIn-like" channels and 4 "VIn-like" channels, each L3→L2
  air.channel @chan_a_0 [1]
  air.channel @chan_a_1 [1]
  air.channel @chan_a_2 [1]
  air.channel @chan_a_3 [1]
  air.channel @chan_b_0 [1]
  air.channel @chan_b_1 [1]
  air.channel @chan_b_2 [1]
  air.channel @chan_b_3 [1]
  // Output channel
  air.channel @chan_out [4]

  func.func @test_sequential_packing(%in_a: memref<256xbf16>, %in_b: memref<256xbf16>, %out: memref<256xbf16>) {
    air.launch () in () args(%arg_a=%in_a, %arg_b=%in_b, %arg_out=%out) : memref<256xbf16>, memref<256xbf16>, memref<256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c192 = arith.constant 192 : index

      // L3→L2 puts for chan_a (4 channels)
      air.channel.put @chan_a_0[%c0] (%arg_a[%c0] [%c64] [%c1]) {id = 1 : i32} : (memref<256xbf16>)
      air.channel.put @chan_a_1[%c0] (%arg_a[%c64] [%c64] [%c1]) {id = 2 : i32} : (memref<256xbf16>)
      air.channel.put @chan_a_2[%c0] (%arg_a[%c128] [%c64] [%c1]) {id = 3 : i32} : (memref<256xbf16>)
      air.channel.put @chan_a_3[%c0] (%arg_a[%c192] [%c64] [%c1]) {id = 4 : i32} : (memref<256xbf16>)

      // L3→L2 puts for chan_b (4 channels)
      air.channel.put @chan_b_0[%c0] (%arg_b[%c0] [%c64] [%c1]) {id = 5 : i32} : (memref<256xbf16>)
      air.channel.put @chan_b_1[%c0] (%arg_b[%c64] [%c64] [%c1]) {id = 6 : i32} : (memref<256xbf16>)
      air.channel.put @chan_b_2[%c0] (%arg_b[%c128] [%c64] [%c1]) {id = 7 : i32} : (memref<256xbf16>)
      air.channel.put @chan_b_3[%c0] (%arg_b[%c192] [%c64] [%c1]) {id = 8 : i32} : (memref<256xbf16>)

      // L2→L3 gets for output
      air.channel.get @chan_out[%c0] (%arg_out[%c0] [%c64] [%c1]) {id = 9 : i32} : (memref<256xbf16>)
      air.channel.get @chan_out[%c1] (%arg_out[%c64] [%c64] [%c1]) {id = 10 : i32} : (memref<256xbf16>)

      air.segment @seg attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index

        // L2 buffers — one per channel, allocated on different memtiles
        %l2_a0 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_a1 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_a2 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_a3 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_b0 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_b1 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_b2 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_b3 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_out0 = memref.alloc() : memref<64xbf16, 1 : i32>
        %l2_out1 = memref.alloc() : memref<64xbf16, 1 : i32>

        // L3→L2 gets
        air.channel.get @chan_a_0[%c0_s] (%l2_a0[] [] []) {id = 11 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_a_1[%c0_s] (%l2_a1[] [] []) {id = 12 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_a_2[%c0_s] (%l2_a2[] [] []) {id = 13 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_a_3[%c0_s] (%l2_a3[] [] []) {id = 14 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_b_0[%c0_s] (%l2_b0[] [] []) {id = 15 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_b_1[%c0_s] (%l2_b1[] [] []) {id = 16 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_b_2[%c0_s] (%l2_b2[] [] []) {id = 17 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.get @chan_b_3[%c0_s] (%l2_b3[] [] []) {id = 18 : i32} : (memref<64xbf16, 1 : i32>)

        // L2→L3 puts for output
        air.channel.put @chan_out[%c0_s] (%l2_out0[] [] []) {id = 19 : i32} : (memref<64xbf16, 1 : i32>)
        air.channel.put @chan_out[%c1_s] (%l2_out1[] [] []) {id = 20 : i32} : (memref<64xbf16, 1 : i32>)

        air.herd @herd tile (%tx, %ty) in (%htx=%c4, %hty=%c1_s)
            attributes {id = 3 : i32} {
          %l1_buf = memref.alloc() : memref<64xbf16, 2 : i32>
          memref.dealloc %l1_buf : memref<64xbf16, 2 : i32>
        }

        memref.dealloc %l2_a0 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_a1 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_a2 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_a3 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_b0 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_b1 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_b2 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_b3 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_out0 : memref<64xbf16, 1 : i32>
        memref.dealloc %l2_out1 : memref<64xbf16, 1 : i32>
      }
    }
    return
  }
}
