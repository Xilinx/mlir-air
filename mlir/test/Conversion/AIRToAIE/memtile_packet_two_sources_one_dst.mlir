//===- memtile_packet_two_sources_one_dst.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Regression for the memtile packet-flow OVER-COLLAPSE bug in
// MemTileDMAAllocator::simpleDmaChannelAlloc. The packet-reuse branch matched
// only the destination tile (foundPacketFlowAllocInTile), so N distinct packet
// flows landing on the same receiver memtile were collapsed onto ONE physical
// S2MM channel regardless of source. That is wrong on the S2MM (receiver) side:
// each distinct (channel symbol, indices) endpoint is a different logical flow
// / source and must get its own physical S2MM channel; only multiple invocations
// of the SAME flow (scf.for unroll or ping-pong) may share one channel.
//
// Here two distinct packet channels (@w0, @w1) each write a disjoint sub-region
// of one L2 buffer at the receiver memtile. With source discrimination the
// memtile allocates TWO S2MM channels (one per source); without it both collapse
// onto S2MM 0. The MM2S (source) side keeps promiscuous collapse (used by
// broadcast packet fan-out and multi-pkt_id multiplexing), so it is unaffected.

// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%{{.*}} : memref<2x8xbf16, 1>, 0, 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// CHECK: aie.dma_start(S2MM, 1
// CHECK: aie.dma_bd(%{{.*}} : memref<2x8xbf16, 1>, 8, 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>

air.channel @w0 [1, 1] {channel_type = "npu_dma_packet"}
air.channel @w1 [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @memtile_pkt_two_src() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      // Two distinct packet channels write disjoint rows of the same L2 buffer.
      air.channel.get @w0[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<2x8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0) attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0) attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<16xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<16xbf16, 2>}
      }
    }
  }
  return
}
