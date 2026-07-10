//===- packet_multi_id_demux.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" | FileCheck %s

// Per-destination demux: a broadcast packet channel with N pinned packet_ids
// fanning out to N distinct receivers routes destination i with pinned id i.
// This gives each receiver its own statically-known routing id instead of a
// single shared broadcast id, so a switchbox arbiter can steer each slice to
// the right receiver by id.
//
// Without the pinned-id path a broadcast is one flow with a single shared id
// fanning to every dest, so the per-receiver routing id is lost.

// One source, two receivers, distinct pinned ids 1 and 4 (one per dest).
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT: aie.packet_source<%[[SRC:.*]], DMA : 0>
// CHECK-NEXT: aie.packet_dest<%[[D0:.*]], DMA : 0>
// CHECK: aie.packet_flow(4) {
// CHECK-NEXT: aie.packet_source<%[[SRC]], DMA : 0>
// CHECK-NEXT: aie.packet_dest<%[[D1:.*]], DMA : 0>

#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @bc [1, 1] {broadcast_shape = [2, 1], channel_type = "npu_dma_packet", packet_ids = [1, 4]}
  func.func @demux() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%c=%c1, %d=%c1) {
      air.segment @seg attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c0_0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
          %aa = memref.alloc() : memref<8xbf16, 1>
          air.execute_terminator %aa : memref<8xbf16, 1>
        }
        air.channel.put @bc[%c0_0, %c0_0] (%l2[] [] []) : (memref<8xbf16, 1>)
        %dd = air.execute { memref.dealloc %l2 : memref<8xbf16, 1> }
        air.herd @h tile (%tx, %ty) in (%sx=%c2_0, %sy=%c1_0)
              attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
            %bb = memref.alloc() : memref<8xbf16, 2>
            air.execute_terminator %bb : memref<8xbf16, 2>
          }
          %g = affine.if #set0()[%tx, %ty] -> !air.async.token {
            %x = air.channel.get async @bc[%tx, %ty] (%l1[] [] []) : (memref<8xbf16, 2>)
            affine.yield %x : !air.async.token
          } else {
            %x = air.channel.get async @bc[%tx, %ty] (%l1[] [] []) : (memref<8xbf16, 2>)
            affine.yield %x : !air.async.token
          }
          %dl = air.execute [%g] { memref.dealloc %l1 : memref<8xbf16, 2> }
        }
      }
    }
    return
  }
}
