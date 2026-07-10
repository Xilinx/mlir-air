//===- packet_multi_id_one_dest.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Intermediate-hop packet routing: a channel with N pinned packet_ids and a
// SINGLE destination emits N packet_flows (one per id) to that one dest. Every
// id routes to the same buffer; a later demux hop keys off the header carried
// by the payload. (Contrast per-destination demux, where N ids fan out to N
// distinct dests.)
//
// Without the pinned-id path, the channel's packet_ids attribute is ignored:
// the auto-assigner hands the flow a SINGLE id, so only ONE packet_flow is
// emitted and the second routing id is lost.

// CHECK: aie.device

// Two flows, SAME source + SAME dest, distinct pinned ids 1 and 4.
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT: aie.packet_source<%[[SRC:.*]], DMA : 0>
// CHECK-NEXT: aie.packet_dest<%[[DST:.*]], DMA : 0>
// CHECK: aie.packet_flow(4) {
// CHECK-NEXT: aie.packet_source<%[[SRC]], DMA : 0>
// CHECK-NEXT: aie.packet_dest<%[[DST]], DMA : 0>

air.channel @m [1, 1] {channel_type = "npu_dma_packet", packet_ids = [1, 4]}
func.func @multi_id_one_dest() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %c66 = arith.constant 66 : index
      // Single destination memtile buffer.
      %t, %l2 = air.execute -> (memref<66xbf16, 1>) {
        %alloc = memref.alloc() : memref<66xbf16, 1>
        air.execute_terminator %alloc : memref<66xbf16, 1>
      }
      air.channel.get @m[] (%l2[] [] []) : (memref<66xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<66xbf16, 1>
      }
      // Producer core: one put feeding both routing ids.
      air.herd @hp tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %ch1 = arith.constant 1 : index
        %ch14 = arith.constant 14 : index
        %ch66 = arith.constant 66 : index
        %tok, %l1 = air.execute -> (memref<80xbf16, 2>) {
          %aa = memref.alloc() : memref<80xbf16, 2>
          air.execute_terminator %aa : memref<80xbf16, 2>
        }
        air.channel.put @m[] (%l1[%ch14] [%ch66] [%ch1]) : (memref<80xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<80xbf16, 2>}
      }
    }
  }
  return
}
