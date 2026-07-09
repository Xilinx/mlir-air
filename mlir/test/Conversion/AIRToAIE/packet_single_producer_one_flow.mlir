//===- packet_single_producer_one_flow.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Single-producer regression pin for the multi-producer change: a packet
// channel @x with ONE producer must still lower to EXACTLY ONE packet_flow
// (numMM2SAllocs == 1, "lowers identically"). Guards against the per-producer
// vectorization accidentally emitting spurious extra flows for the common
// 1-producer case.

// EXACTLY one packet_flow (one producer -> one memtile S2MM):
// CHECK-COUNT-1: aie.packet_flow({{[0-9]+}}) {
// CHECK-NOT: aie.packet_flow(

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @single_producer_one_flow() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      air.channel.get @x[] (%l2[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<8xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<8xbf16, 1> }
      // Single producer herd.
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
