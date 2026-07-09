//===- memtile_packet_same_flow_collapse.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Companion to memtile_packet_two_sources_one_dst.mlir. Verifies the
// collapse-PRESERVING side of the S2MM discrimination in
// MemTileDMAAllocator::simpleDmaChannelAlloc: multiple invocations of the SAME
// logical flow (same channel decl + same indices, e.g. scf.for unroll or
// ping-pong) still time-multiplex ONE physical S2MM channel. Only distinct
// sources get separate channels -- same-flow invocations must not.
//
// Here one packet channel (@w0) is used by two get invocations writing disjoint
// rows of one L2 buffer. Both share S2MM 0 (two BDs, one pkt_id); no second
// S2MM channel is allocated.

// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%{{.*}} : memref<2x8xbf16, 1>, 0, 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// CHECK: aie.dma_bd(%{{.*}} : memref<2x8xbf16, 1>, 8, 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// CHECK-NOT: aie.dma_start(S2MM, 1

air.channel @w0 [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @memtile_pkt_same_flow() {
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
      // Two invocations of the SAME channel write disjoint rows of one buffer.
      air.channel.get @w0[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @w0[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<2x8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        %tok1, %l1b = air.execute -> (memref<8xbf16, 2>) {
          %bb = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %bb : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1b[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
        %d0b = air.execute {memref.dealloc %l1b : memref<8xbf16, 2>}
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
