//===- packet_producers_mixed_memspace.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Multi-producer packet convergence with MIXED-memory-space producers: one
// packet channel @x is fed by BOTH an L2 memtile (a relay buffer streamed back
// out MM2S) and an L1 compute core, converging on one destination memtile S2MM.
// This exercises the per-producer DMA-resource dispatch: an L2-memtile producer
// gets a memtile MM2S, an L1-core producer gets a tile MM2S, both landing on the
// same destination S2MM with the same pkt id.
//
// Before the multi-producer change (scalar MM2S_alloc), this pattern did not
// lower at all: the L2-memtile packet producer hit "memcpy op not outlined in
// an aie.core op" because the single-producer model assumed a core source.

// A memtile-sourced flow AND a core-sourced flow, SAME pkt id, converging on the
// SAME destination memtile S2MM:
// CHECK: aie.packet_flow([[PID:[0-9]+]]) {
// CHECK-NEXT: aie.packet_source<%logical_mem, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST:.*]], DMA : [[DCH:[0-9]+]]>
// CHECK: aie.packet_flow([[PID]]) {
// CHECK-NEXT: aie.packet_source<%tile_{{[0-9]+_[0-9]+}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST]], DMA : [[DCH]]>

air.channel @tomt [1, 1]
air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @mixed_producers() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      // L2 memtile producer: relay a core-produced buffer onto @x (MM2S from L2).
      %tm, %l2p = air.execute -> (memref<8xbf16, 1>) {
        %a2 = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %a2 : memref<8xbf16, 1>
      }
      air.channel.get @tomt[] (%l2p[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @x[] (%l2p[] [] []) : (memref<8xbf16, 1>)
      // Destination memtile buffer assembled from @x (2 producers: L2 + L1).
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      air.channel.get @x[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<2x8xbf16, 1> }
      %dp = air.execute { memref.dealloc %l2p : memref<8xbf16, 1> }
      // L1 core producer #1: feeds the L2 relay via @tomt.
      air.herd @hp tile (%txp, %typ) in (%sxp=%c1_0, %syp=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @tomt[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dpp = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      // L1 core producer #2: puts directly onto @x (the mixed 2nd producer).
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
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
