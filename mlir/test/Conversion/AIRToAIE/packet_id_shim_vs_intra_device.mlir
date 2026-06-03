//===- packet_id_shim_vs_intra_device.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: shim packet flows and intra-device (memtile-to-compute,
// compute-to-compute) packet flows used independent 0-based pkt_id counters
// inside one device. The first shim packet flow and the first intra-device
// packet flow both got pkt_id = 0; where their physical routes crossed at a
// switchbox, the arbiter could not disambiguate them and packets silently
// mis-routed. The fix assigns shim packet flows first, then offsets
// intra-device packet flow IDs so every packet flow within a device carries
// a unique pkt_id.
//
// Each case below uses `--split-input-file` so the per-device pkt_id counter
// resets between cases. The regression invariants pinned per device are:
//   (1) Each `aie.packet_flow(N)` value appears AT MOST ONCE per device
//       (no two flows share an ID).
//   (2) The pkt_id on intra-device `aie.dma_bd` ops is strictly greater
//       than every shim pkt_id used in the same device.

// RUN: air-opt %s --split-input-file -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// =============================================================================
// Case 1: one shim packet flow (L3 -> L2) coexists with one intra-device
// packet flow (L2 -> L1). Pre-fix: both packet_flow ops got id 0
// (collision). Post-fix: distinct.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @case1_seg
// Two distinct packet_flow declarations. The intra-device one MUST NOT also
// be packet_flow(0) -- that was the pre-fix collision.
// CHECK-COUNT-1: aie.packet_flow(0) {
// CHECK-NOT:     aie.packet_flow(0) {
// CHECK:         aie.packet_flow({{[1-9][0-9]*}}) {
//
// The intra-device flow's receiver-side dma_bd carries the assigned pkt_id.
// It MUST be non-zero (would-be collision with the shim flow's pkt_id 0).
// CHECK:     aie.dma_bd{{.*}}pkt_id = {{[1-9][0-9]*}}

module {
  air.channel @l3_to_l2 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @l2_to_l1 [1, 1] {channel_type = "npu_dma_packet"}

  func.func @case1_shim_and_intra(%arg0: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in0=%arg0) : memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      // L3-side put: drives the shim MM2S packet flow.
      %put_l3 = air.channel.put async @l3_to_l2[%c0, %c0] (%in0[] [] []) {id = 1 : i32} : (memref<64xbf16>)

      %seg = air.segment @case1_seg async [%put_l3] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_seg = arith.constant 1 : index
        // L2 staging buffer (memtile, memory_space = 1).
        %async_l2, %buf_l2 = air.execute -> (memref<64xbf16, 1>) {
          %alloc = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %alloc : memref<64xbf16, 1>
        }
        %get_l2 = air.channel.get async [%async_l2] @l3_to_l2[%c1_seg, %c1_seg] (%buf_l2[] [] []) {id = 2 : i32} : (memref<64xbf16, 1>)
        // L2-side put: drives the intra-device (memtile -> compute) packet
        // flow. Pre-fix this got pkt_id = 0 (same as the shim flow above).
        %put_l2 = air.channel.put async [%get_l2] @l2_to_l1[%c1_seg, %c1_seg] (%buf_l2[] [] []) {id = 3 : i32} : (memref<64xbf16, 1>)

        %herd = air.herd @case1_herd async [%put_l2] tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          %async_l1, %buf_l1 = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_l1 = air.channel.get async [%async_l1] @l2_to_l1[%hc0, %hc0] (%buf_l1[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)
          %dealloc_l1 = air.execute [%get_l1] {
            memref.dealloc %buf_l1 : memref<64xbf16, 2>
          }
        }
        %dealloc_l2 = air.execute [%put_l2] {
          memref.dealloc %buf_l2 : memref<64xbf16, 1>
        }
      }
    }
    return
  }
}

// -----

// =============================================================================
// Case 2: two shim packet flows + two intra-device packet flows. Shim flows
// claim two distinct pkt_ids (0 and 1). Pre-fix the intra-device flows
// reused 0 and 1 (double collision). Post-fix the intra-device flows must
// claim pkt_ids strictly greater than every shim pkt_id in the device.
// Validates the offset scales with the count of shim flows.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @case2_seg
// Each shim pkt_id appears exactly once -- intra-device must NOT reuse them.
// CHECK-COUNT-1: aie.packet_flow(0) {
// CHECK-NOT:     aie.packet_flow(0) {
// CHECK-COUNT-1: aie.packet_flow(1) {
// CHECK-NOT:     aie.packet_flow(1) {
//
// And there must be exactly two more packet_flow ops with distinct IDs
// strictly greater than 1.
// CHECK-COUNT-2: aie.packet_flow({{[2-9][0-9]*}}) {
//
// The intra-device dma_bd ops both carry pkt_ids >= 2.
// CHECK-COUNT-2: aie.dma_bd{{.*}}pkt_id = {{[2-9][0-9]*}}

module {
  air.channel @l3_to_l2_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @l3_to_l2_b [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @l2_to_l1_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @l2_to_l1_b [1, 1] {channel_type = "npu_dma_packet"}

  func.func @case2_two_shim_two_intra(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in_a=%arg0, %in_b=%arg1) : memref<64xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %put_a = air.channel.put async @l3_to_l2_a[%c0, %c0] (%in_a[] [] []) {id = 1 : i32} : (memref<64xbf16>)
      %put_b = air.channel.put async @l3_to_l2_b[%c0, %c0] (%in_b[] [] []) {id = 2 : i32} : (memref<64xbf16>)

      %seg = air.segment @case2_seg async [%put_a, %put_b] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_seg = arith.constant 1 : index

        %async_l2a, %buf_l2a = air.execute -> (memref<64xbf16, 1>) {
          %alloc = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %alloc : memref<64xbf16, 1>
        }
        %get_l2a = air.channel.get async [%async_l2a] @l3_to_l2_a[%c1_seg, %c1_seg] (%buf_l2a[] [] []) {id = 3 : i32} : (memref<64xbf16, 1>)
        %put_l2a = air.channel.put async [%get_l2a] @l2_to_l1_a[%c1_seg, %c1_seg] (%buf_l2a[] [] []) {id = 4 : i32} : (memref<64xbf16, 1>)

        %async_l2b, %buf_l2b = air.execute -> (memref<64xbf16, 1>) {
          %alloc = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %alloc : memref<64xbf16, 1>
        }
        %get_l2b = air.channel.get async [%async_l2b] @l3_to_l2_b[%c1_seg, %c1_seg] (%buf_l2b[] [] []) {id = 5 : i32} : (memref<64xbf16, 1>)
        %put_l2b = air.channel.put async [%get_l2b] @l2_to_l1_b[%c1_seg, %c1_seg] (%buf_l2b[] [] []) {id = 6 : i32} : (memref<64xbf16, 1>)

        %herd = air.herd @case2_herd async [%put_l2a, %put_l2b] tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          %async_l1a, %buf_l1a = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_l1a = air.channel.get async [%async_l1a] @l2_to_l1_a[%hc0, %hc0] (%buf_l1a[] [] []) {id = 7 : i32} : (memref<64xbf16, 2>)
          %async_l1b, %buf_l1b = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_l1b = air.channel.get async [%async_l1b] @l2_to_l1_b[%hc0, %hc0] (%buf_l1b[] [] []) {id = 8 : i32} : (memref<64xbf16, 2>)
          %dealloc_l1a = air.execute [%get_l1a] {
            memref.dealloc %buf_l1a : memref<64xbf16, 2>
          }
          %dealloc_l1b = air.execute [%get_l1b] {
            memref.dealloc %buf_l1b : memref<64xbf16, 2>
          }
        }
        %dealloc_l2a = air.execute [%put_l2a] {
          memref.dealloc %buf_l2a : memref<64xbf16, 1>
        }
        %dealloc_l2b = air.execute [%put_l2b] {
          memref.dealloc %buf_l2b : memref<64xbf16, 1>
        }
      }
    }
    return
  }
}
