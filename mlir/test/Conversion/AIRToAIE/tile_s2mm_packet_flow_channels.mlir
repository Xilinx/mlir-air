//===- tile_s2mm_packet_flow_channels.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" -split-input-file | FileCheck %s

// Two inbound packet flows into one tile, both received in a loop. Their
// producers are independent, so a shared S2MM BD ring would take their
// arrivals in no fixed order. Each gets its own channel.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%{{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// CHECK: aie.next_bd
// CHECK: aie.dma_start(S2MM, 1
// CHECK: aie.dma_bd(%{{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>
// CHECK: aie.next_bd

air.channel @fX [1] {channel_type = "npu_dma_packet"}
air.channel @fR [1] {channel_type = "npu_dma_packet"}
func.func @both_repeat(%x: memref<64x32xbf16>, %r: memref<64x32xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%ax=%x, %ar=%r) : memref<64x32xbf16>, memref<64x32xbf16> {
    air.segment @seg args(%sx=%ax, %sr=%ar) : memref<64x32xbf16>, memref<64x32xbf16> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0_0 to %c4 step %c1_0 {
        air.channel.put @fX[] (%sx[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 1 : i32} : (memref<64x32xbf16>)
        air.channel.put @fR[] (%sr[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 2 : i32} : (memref<64x32xbf16>)
      }
      air.herd @h tile (%tx, %ty) in (%sx2=%c1_0, %sy2=%c1_0) {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c4_1 = arith.constant 4 : index
        %bx = memref.alloc() : memref<1x8xbf16, 2>
        %br = memref.alloc() : memref<1x8xbf16, 2>
        scf.for %j = %c0_1 to %c4_1 step %c1_1 {
          air.channel.get @fX[] (%bx[] [] []) {id = 3 : i32} : (memref<1x8xbf16, 2>)
          air.channel.get @fR[] (%br[] [] []) {id = 4 : i32} : (memref<1x8xbf16, 2>)
        }
        memref.dealloc %bx : memref<1x8xbf16, 2>
        memref.dealloc %br : memref<1x8xbf16, 2>
      }
    }
  }
  return
}

// -----

// One flow received once, the other in a loop: they still share S2MM 0.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NOT: aie.dma_start(S2MM, 1
// CHECK: aie.end

air.channel @gW [1] {channel_type = "npu_dma_packet"}
air.channel @gX [1] {channel_type = "npu_dma_packet"}
func.func @one_repeats(%w: memref<8xbf16>, %x: memref<64x32xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%aw=%w, %ax=%x) : memref<8xbf16>, memref<64x32xbf16> {
    air.segment @seg args(%sw=%aw, %sx=%ax) : memref<8xbf16>, memref<64x32xbf16> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      air.channel.put @gW[] (%sw[] [] []) {id = 1 : i32} : (memref<8xbf16>)
      scf.for %i = %c0_0 to %c4 step %c1_0 {
        air.channel.put @gX[] (%sx[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 2 : i32} : (memref<64x32xbf16>)
      }
      air.herd @h tile (%tx, %ty) in (%sx2=%c1_0, %sy2=%c1_0) {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c4_1 = arith.constant 4 : index
        %bw = memref.alloc() : memref<8xbf16, 2>
        %bx = memref.alloc() : memref<1x8xbf16, 2>
        air.channel.get @gW[] (%bw[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        scf.for %j = %c0_1 to %c4_1 step %c1_1 {
          air.channel.get @gX[] (%bx[] [] []) {id = 4 : i32} : (memref<1x8xbf16, 2>)
        }
        memref.dealloc %bw : memref<8xbf16, 2>
        memref.dealloc %bx : memref<1x8xbf16, 2>
      }
    }
  }
  return
}

// -----

// Both flows are also received once before the loop. The flow still repeats,
// so the first receive must not put it on the other flow's ring.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM, 0
// CHECK: pkt_id = 0
// CHECK: aie.dma_start(S2MM, 1
// CHECK: pkt_id = 1

air.channel @pAA [1] {channel_type = "npu_dma_packet"}
air.channel @pAB [1] {channel_type = "npu_dma_packet"}
func.func @prologue_and_loop(%a: memref<64x32xbf16>, %b: memref<64x32xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %ab=%b) : memref<64x32xbf16>, memref<64x32xbf16> {
    air.segment @seg args(%sa=%aa, %sb=%ab) : memref<64x32xbf16>, memref<64x32xbf16> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0_0 to %c5 step %c1_0 {
        air.channel.put @pAA[] (%sa[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 1 : i32} : (memref<64x32xbf16>)
        air.channel.put @pAB[] (%sb[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 2 : i32} : (memref<64x32xbf16>)
      }
      air.herd @h tile (%tx, %ty) in (%sx2=%c1_0, %sy2=%c1_0) {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c4_1 = arith.constant 4 : index
        %ba = memref.alloc() : memref<1x8xbf16, 2>
        %bb = memref.alloc() : memref<1x8xbf16, 2>
        air.channel.get @pAA[] (%ba[] [] []) {id = 6 : i32} : (memref<1x8xbf16, 2>)
        air.channel.get @pAB[] (%bb[] [] []) {id = 7 : i32} : (memref<1x8xbf16, 2>)
        scf.for %j = %c0_1 to %c4_1 step %c1_1 {
          air.channel.get @pAA[] (%ba[] [] []) {id = 4 : i32} : (memref<1x8xbf16, 2>)
          air.channel.get @pAB[] (%bb[] [] []) {id = 5 : i32} : (memref<1x8xbf16, 2>)
        }
        memref.dealloc %ba : memref<1x8xbf16, 2>
        memref.dealloc %bb : memref<1x8xbf16, 2>
      }
    }
  }
  return
}

// -----

// One flow is pinned to S2MM 1. The other repeating flow takes the free S2MM 0
// rather than the round-robin's channel 1.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM, 1
// CHECK: pkt_id = 0
// CHECK: aie.dma_start(S2MM, 0
// CHECK: pkt_id = 1

air.channel @pnA [1] {channel_type = "npu_dma_packet", air.tile_dma_channel = 1 : i32}
air.channel @pnB [1] {channel_type = "npu_dma_packet"}
func.func @one_pinned(%a: memref<64x32xbf16>, %b: memref<64x32xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %ab=%b) : memref<64x32xbf16>, memref<64x32xbf16> {
    air.segment @seg args(%sa=%aa, %sb=%ab) : memref<64x32xbf16>, memref<64x32xbf16> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0_0 to %c4 step %c1_0 {
        air.channel.put @pnA[] (%sa[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 1 : i32} : (memref<64x32xbf16>)
        air.channel.put @pnB[] (%sb[%i, %c0_0] [%c1_0, %c8] [%c32, %c1_0]) {id = 2 : i32} : (memref<64x32xbf16>)
      }
      air.herd @h tile (%tx, %ty) in (%sx2=%c1_0, %sy2=%c1_0) {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c4_1 = arith.constant 4 : index
        %ba = memref.alloc() : memref<1x8xbf16, 2>
        %bb = memref.alloc() : memref<1x8xbf16, 2>
        scf.for %j = %c0_1 to %c4_1 step %c1_1 {
          air.channel.get @pnA[] (%ba[] [] []) {id = 4 : i32} : (memref<1x8xbf16, 2>)
          air.channel.get @pnB[] (%bb[] [] []) {id = 5 : i32} : (memref<1x8xbf16, 2>)
        }
        memref.dealloc %ba : memref<1x8xbf16, 2>
        memref.dealloc %bb : memref<1x8xbf16, 2>
      }
    }
  }
  return
}
