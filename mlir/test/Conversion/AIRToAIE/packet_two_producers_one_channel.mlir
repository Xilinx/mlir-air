//===- packet_two_producers_one_channel.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// N-producer -> 1-consumer PACKET convergence: a broadcast/assembly buffer
// is fed by multiple producer cores writing ONE memtile S2MM via packet flows.
// Distinct from
// memtile_packet_two_sources_one_dst.mlir, which uses N DISTINCT channels (each
// single-producer); here it is ONE channel @x with TWO puts from TWO distinct
// producer tiles.
//
// The bug: `MemcpyBundleAsFlow` modelled a flow as 1-producer -> N-consumer
// (scalar MM2S_alloc), so the two same-channel puts collapsed to one alloc and
// air-to-aie emitted only ONE packet_flow -- the second producer's flow was
// silently DROPPED, so the consumer slot it fed never filled (runtime deadlock).
// Fix: MM2S_alloc is a vector (one per distinct producer tile), so a multi-put
// packet channel emits one packet_flow per producer, all converging on the
// shared destination S2MM with the SAME pkt id (foundPacketFlowAllocInTile
// merges the dest).

// Both producer tiles exist:
// CHECK-DAG: aie.tile(2, 3)
// CHECK-DAG: aie.tile(3, 3)
// TWO packet_flows (one per producer), carrying the SAME pkt id and converging
// on the SAME memtile destination DMA channel. Pre-fix only ONE was emitted
// (the second producer's flow was dropped).
// CHECK: aie.packet_flow([[PID:[0-9]+]]) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[MT:.*]], DMA : [[DCH:[0-9]+]]>
// CHECK: aie.packet_flow([[PID]]) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[MT]], DMA : [[DCH]]>

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @packet_two_producers_one_channel() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      // Shared L2 buffer fed by the two producers (one sub-region row each).
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      // TWO gets from the SAME channel @x -> converge on one memtile S2MM.
      air.channel.get @x[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c1_1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<2x8xbf16, 1>
      }
      // TWO producer herds on DISTINCT tiles, each putting to @x.
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      // Consumer herd reads the assembled buffer.
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
