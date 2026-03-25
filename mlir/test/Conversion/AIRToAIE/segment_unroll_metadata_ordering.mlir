//===- segment_unroll_metadata_ordering.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that metadataArray entries use per-device tile indices and are correctly
// reordered for segment unroll. Two cases are tested:
// 1) 1D channel [2] with 2x1 unroll — per-device naming only (no sort needed)
// 2) 2D channel [2, 2] with 2x1 unroll — sort code reorders metadataArray

// RUN: air-opt %s -split-input-file -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s --check-prefixes=CHECK1D,CHECK2D

// ============================================================
// Test 1: 1D channel with segment unroll (per-device naming)
// ============================================================

// The output channel @out_chan has size=[2] (one per segment unroll copy).
// After segment unroll, shim allocations are created for each (tile, unroll).
// The trailing index in each name is the per-device tile index (0-based
// within each device), not the globally-sequential metadataArray position.

// Check that both devices are created with correct allocations:
// CHECK1D-LABEL: aie.device{{.*}}@segment_meta_0_0
// CHECK1D:       aie.shim_dma_allocation @air_out_chan_0_0
// CHECK1D:       segment_unroll_x = 0

// CHECK1D-LABEL: aie.device{{.*}}@segment_meta_1_0
// CHECK1D:       aie.shim_dma_allocation @air_out_chan_1_0_0
// CHECK1D:       segment_unroll_x = 1

// Check metadataArray ordering on the launch-body channel gets.
// CHECK1D: air.channel.get @out_chan[%c0]
// CHECK1D-SAME: metadataArray = [{base = "air_out_chan_0_0"
// CHECK1D-SAME:                   {base = "air_out_chan_1_0_0"
// CHECK1D: air.channel.get @out_chan[%c1]
// CHECK1D-SAME: metadataArray = [{base = "air_out_chan_0_0"
// CHECK1D-SAME:                   {base = "air_out_chan_1_0_0"

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

// -----

// ============================================================
// Test 2: 2D channel with segment unroll (sort code exercised)
// ============================================================

// The output channel @out_2d has size=[2, 2] (2 tiles × 2 unroll heads).
// With 2x1 segment unroll, each device produces 2 tiles of output.
// This creates 4 shim allocations total (2 per device).
//
// Without the per-device index fix, device 1's allocations would have
// tileIdx=2,3 in their names (global t_idx), causing the sort code to
// compute out-of-bounds linearized indices and silently skip sorting.
//
// With the fix, device 1's allocations have tileIdx=0,1 (per-device),
// and the sort code correctly reorders the metadataArray so that
// getIteratorFromMDVector({2,2}, {col, head}) maps to the right entry:
//   [0,0]→pos 0 (dev0 tile0), [0,1]→pos 1 (dev1 tile0),
//   [1,0]→pos 2 (dev0 tile1), [1,1]→pos 3 (dev1 tile1)

// Check per-device shim allocations have per-device tile indices:
// CHECK2D-LABEL: aie.device{{.*}}@segment_2d_0_0
// CHECK2D-DAG:   aie.shim_dma_allocation @air_out_2d_0_0_0(%{{.*}}, S2MM, 0)
// CHECK2D-DAG:   aie.shim_dma_allocation @air_out_2d_0_0_1(%{{.*}}, S2MM, 1)

// CHECK2D-LABEL: aie.device{{.*}}@segment_2d_1_0
// CHECK2D-DAG:   aie.shim_dma_allocation @air_out_2d_1_0_0(%{{.*}}, S2MM, 0)
// CHECK2D-DAG:   aie.shim_dma_allocation @air_out_2d_1_0_1(%{{.*}}, S2MM, 1)

// Check that the metadataArray is sorted to match getIteratorFromMDVector
// linearization: position = {tileIdx, unrollCopy}, dims = {2, 2}.
//   linIdx 0 = {0,0} → dev0 tile0 (air_out_2d_0_0_0)
//   linIdx 1 = {0,1} → dev1 tile0 (air_out_2d_1_0_0)
//   linIdx 2 = {1,0} → dev0 tile1 (air_out_2d_0_0_1)
//   linIdx 3 = {1,1} → dev1 tile1 (air_out_2d_1_0_1)
// CHECK2D: air.channel.get @out_2d[%c0, %c0]
// CHECK2D-SAME: metadataArray = [{base = "air_out_2d_0_0_0"
// CHECK2D-SAME:                   {base = "air_out_2d_1_0_0"
// CHECK2D-SAME:                   {base = "air_out_2d_0_0_1"
// CHECK2D-SAME:                   {base = "air_out_2d_1_0_1"

module {
  air.channel @out_2d [2, 2]

  func.func @test_2d_metadata_sort(%arg0: memref<256xbf16>, %out: memref<256xbf16>) {
    %0 = air.launch async () in () args(%input=%arg0, %output=%out) : memref<256xbf16>, memref<256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c192 = arith.constant 192 : index

      // Output channel gets at launch level (L2→L3)
      // 4 gets for @out_2d[col, head] where col=0..1, head=0..1
      air.channel.get @out_2d[%c0, %c0] (%output[%c0] [%c64] [%c1]) {id = 10 : i32} : (memref<256xbf16>)
      air.channel.get @out_2d[%c1, %c0] (%output[%c64] [%c64] [%c1]) {id = 11 : i32} : (memref<256xbf16>)
      air.channel.get @out_2d[%c0, %c1] (%output[%c128] [%c64] [%c1]) {id = 12 : i32} : (memref<256xbf16>)
      air.channel.get @out_2d[%c1, %c1] (%output[%c192] [%c64] [%c1]) {id = 13 : i32} : (memref<256xbf16>)

      // 2x1 segment unroll: unroll dim maps to the second channel index
      %segment = air.segment @segment_2d async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1)
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 2 : i64} {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index

        // L2 output buffers — one per tile column
        %l2_out_0 = memref.alloc() : memref<64xbf16, 1>
        %l2_out_1 = memref.alloc() : memref<64xbf16, 1>

        // Output channel puts: 2 tiles per device, unroll index selects head
        air.channel.put @out_2d[%c0_s, %ux] (%l2_out_0[] [] []) {id = 14 : i32} : (memref<64xbf16, 1>)
        air.channel.put @out_2d[%c1_s, %ux] (%l2_out_1[] [] []) {id = 15 : i32} : (memref<64xbf16, 1>)

        %herd = air.herd @herd_2d async tile (%tx, %ty) in (%htx=%c1_s, %hty=%c1_s)
            attributes {id = 3 : i32} {
          %l1_buf = memref.alloc() : memref<64xbf16, 2>
          memref.dealloc %l1_buf : memref<64xbf16, 2>
        }

        memref.dealloc %l2_out_0 : memref<64xbf16, 1>
        memref.dealloc %l2_out_1 : memref<64xbf16, 1>
      }
    }
    return
  }
}
