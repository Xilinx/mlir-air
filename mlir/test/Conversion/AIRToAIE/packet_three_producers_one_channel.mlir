//===- packet_three_producers_one_channel.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// N=3 producer -> 1-consumer PACKET convergence. Extends
// packet_two_producers_one_channel.mlir past N=2 to confirm the per-producer
// grouping loop scales: THREE distinct producer tiles putting to ONE packet
// channel @x emit THREE packet_flows carrying the SAME pkt id and converging on
// the SAME destination memtile S2MM.

// All three producer tiles and the destination exist:
// CHECK-DAG: aie.tile(2, 3)
// CHECK-DAG: aie.tile(3, 3)
// CHECK-DAG: aie.tile(4, 3)
// THREE packet_flows, one per producer, SAME pkt id, SAME destination channel:
// CHECK: aie.packet_flow([[PID:[0-9]+]]) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST:.*]], DMA : [[DCH:[0-9]+]]>
// CHECK: aie.packet_flow([[PID]]) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST]], DMA : [[DCH]]>
// CHECK: aie.packet_flow([[PID]]) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST]], DMA : [[DCH]]>

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @three_producers_one_channel() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      // Shared L2 buffer fed by the three producers (one sub-region row each).
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      // THREE gets from the SAME channel @x -> converge on one memtile S2MM.
      air.channel.get @x[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c2,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<3x8xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<3x8xbf16, 1> }
      // THREE producer herds on DISTINCT tiles, each putting to @x.
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
      air.herd @h2 tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d2 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      // Consumer herd reads the assembled buffer.
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<24xbf16, 2>) {
          %aa = memref.alloc() : memref<24xbf16, 2>
          air.execute_terminator %aa : memref<24xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<24xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<24xbf16, 2>}
      }
    }
  }
  return
}
