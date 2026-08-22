//===- tile_packet_circuit_channel_split.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1_1col" -split-input-file | FileCheck %s

// TileDMAAllocator::spreadCollapsedPacketChannels.
//
// A compute-tile DMA channel is ONE BD ring walked strictly in order, and a
// packet header steers a transfer to the tile without choosing which BD
// receives it. Sharing a ring is therefore safe exactly when something fixes
// the arrival order, and three things can:
//
//   - one switching kind. A switchbox port is statically connected OR
//     packet-arbitrated, never both, and a channel is one port.
//   - a ROUND-ROBIN ring: every flow owns exactly one BD of the repeating
//     unit, so a flow's k-th transfer always meets that same BD however the
//     arrivals interleave.
//   - a single producer, whose own BD order fixes the arrival order.
//
// Absent all three the pass separates the ring; otherwise it leaves it alone,
// because collapse is how the emitter folds a repeating chain into one BD with
// a repeat count (see mm2s_flows_program_order.mlir).
//
// simpleDmaChannelAlloc produces the unsafe cases on its own: it reuses an
// existing packet channel before it looks for a free one, and its fallback
// `chan = num_allocs % tile_dma_channels` counts ALLOCATIONS rather than
// occupied channels, so it hands out a channel already in use while another
// sits idle.

// -----

// The shape above, and the one the llama-3.2-1B q4nx decode rope tile produces:
// @k pinned to MM2S 1, @v collapsing onto it, then the CIRCUIT @q computing
// 1 % 2 = 1 and joining them while MM2S 0 is idle.
//
// @q is what leaves -- it is the flow whose kind differs from the keeper's.
// Evicting @v instead would move a flow that was not part of the problem and
// leave @k sharing with @q. The two packets stay in program order (@k issued
// before @v), which is what the ring delivers in.

// CHECK-LABEL: aie.device
// CHECK:      aie.mem(%[[T:.*]])
// CHECK:        aie.dma_start(MM2S, 1
// CHECK:        aie.dma_bd(%[[K:.*]] :{{.*}}pkt_id = 0
// CHECK:        aie.dma_bd(%[[V:.*]] :{{.*}}pkt_id = 1
// CHECK:        aie.dma_start(MM2S, 0
// CHECK-NOT:    packet
// CHECK:        aie.dma_bd(%{{.*}} : memref<8xbf16, 2> offset = 0 len = 8) {task_id

air.channel @k [1] {air.tile_dma_channel = 1 : i32, channel_type = "npu_dma_packet"}
air.channel @v [1] {channel_type = "npu_dma_packet"}
air.channel @q [1]
func.func @circuit_joins_collapsed_packets(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.segment @seg args(%se=%e) : memref<8xbf16> {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %bk = memref.alloc() : memref<8xbf16, 2>
        %bv = memref.alloc() : memref<8xbf16, 2>
        %bq = memref.alloc() : memref<8xbf16, 2>
        air.channel.put @k[%tx, %ty] (%bk[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.put @v[%tx, %ty] (%bv[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
        air.channel.put @q[%tx, %ty] (%bq[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %bk : memref<8xbf16, 2>
        memref.dealloc %bv : memref<8xbf16, 2>
        memref.dealloc %bq : memref<8xbf16, 2>
      }
      %lk = memref.alloc() : memref<8xbf16, 1>
      %lv = memref.alloc() : memref<8xbf16, 1>
      %lq = memref.alloc() : memref<8xbf16, 1>
      air.channel.get @k[] (%lk[] [] []) {id = 4 : i32} : (memref<8xbf16, 1>)
      air.channel.get @v[] (%lv[] [] []) {id = 5 : i32} : (memref<8xbf16, 1>)
      air.channel.get @q[] (%lq[] [] []) {id = 6 : i32} : (memref<8xbf16, 1>)
      memref.dealloc %lk : memref<8xbf16, 1>
      memref.dealloc %lv : memref<8xbf16, 1>
      memref.dealloc %lq : memref<8xbf16, 1>
    }
  }
  return
}

// -----

// Not a hazard: every flow on the ring packetizes, so one packet-switched port
// carries them all and multiplexing is doing its job. This must stay collapsed
// on ONE channel -- breaking up every shared ring would cost the repeat-count
// fold the emitter gets from it (see mm2s_flows_program_order.mlir).

// CHECK-LABEL: aie.device
// CHECK:      aie.mem
// CHECK:        aie.dma_start(MM2S, 0
// CHECK-NOT:    aie.dma_start(MM2S, 1

air.channel @k3 [1] {channel_type = "npu_dma_packet"}
air.channel @v3 [1] {channel_type = "npu_dma_packet"}
func.func @two_packets_share(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.segment @seg3 args(%se=%e) : memref<8xbf16> {
      %c1_0 = arith.constant 1 : index
      air.herd @h3 tile(%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %bk = memref.alloc() : memref<8xbf16, 2>
        %bv = memref.alloc() : memref<8xbf16, 2>
        air.channel.put @k3[%tx, %ty] (%bk[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.put @v3[%tx, %ty] (%bv[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %bk : memref<8xbf16, 2>
        memref.dealloc %bv : memref<8xbf16, 2>
      }
      %lk = memref.alloc() : memref<8xbf16, 1>
      %lv = memref.alloc() : memref<8xbf16, 1>
      air.channel.get @k3[] (%lk[] [] []) {id = 3 : i32} : (memref<8xbf16, 1>)
      air.channel.get @v3[] (%lv[] [] []) {id = 4 : i32} : (memref<8xbf16, 1>)
      memref.dealloc %lk : memref<8xbf16, 1>
      memref.dealloc %lv : memref<8xbf16, 1>
    }
  }
  return
}

// -----

// `air.tile_dma_channel` still overrides. Both flows are pinned to MM2S 0, one
// packet and one circuit, and the pass leaves them there: the attribute is the
// explicit escape hatch and outranks the rule. It is simply no longer needed to
// obtain the separation in the first case.

// CHECK-LABEL: aie.device
// CHECK:      aie.mem
// CHECK:        aie.dma_start(MM2S, 0
// CHECK-NOT:    aie.dma_start(MM2S, 1

air.channel @q4 [1] {air.tile_dma_channel = 0 : i32}
air.channel @k4 [1] {air.tile_dma_channel = 0 : i32, channel_type = "npu_dma_packet"}
func.func @pin_overrides(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.segment @seg4 args(%se=%e) : memref<8xbf16> {
      %c1_0 = arith.constant 1 : index
      air.herd @h4 tile(%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %bq = memref.alloc() : memref<8xbf16, 2>
        %bk = memref.alloc() : memref<8xbf16, 2>
        air.channel.put @q4[%tx, %ty] (%bq[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.put @k4[%tx, %ty] (%bk[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %bq : memref<8xbf16, 2>
        memref.dealloc %bk : memref<8xbf16, 2>
      }
      %lq = memref.alloc() : memref<8xbf16, 1>
      %lk = memref.alloc() : memref<8xbf16, 1>
      air.channel.get @q4[] (%lq[] [] []) {id = 3 : i32} : (memref<8xbf16, 1>)
      air.channel.get @k4[] (%lk[] [] []) {id = 4 : i32} : (memref<8xbf16, 1>)
      memref.dealloc %lq : memref<8xbf16, 1>
      memref.dealloc %lk : memref<8xbf16, 1>
    }
  }
  return
}
