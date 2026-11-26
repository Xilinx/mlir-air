//===- shim_spread_shared_tile_ring.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -split-input-file | FileCheck %s

// The shim spreads packet flows collapsed onto one MM2S channel over its free
// channels. A compute-tile S2MM ring is walked in order, and a packet does not
// choose its BD: flows from two shim channels arrive in any order, so one ring
// fed by both can put a packet in the other flow's buffer.

// Two flows from the shim into one tile ring. The spread gives @pB its own
// shim channel, and the ring is split by producer: @pB onto the tile's free
// S2MM channel.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(S2MM, 1
// CHECK: aie.shim_dma_allocation @air_pA({{.*}}, MM2S, 0)
// CHECK: aie.shim_dma_allocation @air_pB({{.*}}, MM2S, 1)

air.channel @pA [1] {channel_type = "npu_dma_packet"}
air.channel @pB [1] {channel_type = "npu_dma_packet"}
func.func @split_receiver(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @pA[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @pB[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %a = memref.alloc() : memref<8xbf16, 2>
        %b = memref.alloc() : memref<8xbf16, 2>
        air.channel.get @pA[] (%a[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        air.channel.get @pB[] (%b[] [] []) {id = 4 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %a : memref<8xbf16, 2>
        memref.dealloc %b : memref<8xbf16, 2>
      }
    }
  }
  return
}

// -----

// The same, but the tile's other S2MM channel already takes a flow from
// another core: the ring cannot be split, so the shim keeps @qA and @qB on one
// channel, whose order the ring follows.

// CHECK-LABEL: aie.device
// CHECK: aie.shim_dma_allocation @air_qA({{.*}}, MM2S, 0)
// CHECK: aie.shim_dma_allocation @air_qB({{.*}}, MM2S, 0)

air.channel @qA [1] {channel_type = "npu_dma_packet"}
air.channel @qB [1] {channel_type = "npu_dma_packet"}
air.channel @qC [1]
func.func @no_free_receiver(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @qA[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @qB[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @src tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %s = memref.alloc() : memref<8xbf16, 2>
        air.channel.put @qC[] (%s[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %s : memref<8xbf16, 2>
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %a = memref.alloc() : memref<8xbf16, 2>
        %b = memref.alloc() : memref<8xbf16, 2>
        %c = memref.alloc() : memref<8xbf16, 2>
        air.channel.get @qA[] (%a[] [] []) {id = 4 : i32} : (memref<8xbf16, 2>)
        air.channel.get @qB[] (%b[] [] []) {id = 5 : i32} : (memref<8xbf16, 2>)
        air.channel.get @qC[] (%c[] [] []) {id = 6 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %a : memref<8xbf16, 2>
        memref.dealloc %b : memref<8xbf16, 2>
        memref.dealloc %c : memref<8xbf16, 2>
      }
    }
  }
  return
}
