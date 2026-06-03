//===- recv_packet_filter_demux.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: when two packet flows land on the same receiver L1 DMA
// channel (TileDMAAllocator shares the physical channel via packet-flow
// time multiplexing), each receiver BD must carry its own packet-info
// filter so the hardware DMA can demux. Pre-fix the lookup walked
// PacketFlowOps by source, so receiver-side BDs found nothing; an
// `isMM2S && pktFlowOp` guard then dropped the filter entirely on the
// receiver side, leaving FIFO-order arrivals into whichever ping-pong
// BD was active. This corrupted data and deadlocked the
// rate-imbalanced flow (la_lgu_ld_cascade_fused on NPU2 reproduced).
//
// The fix keys pkt_id off the air.channel symbol name directly via
// packetIDForChannelName, so the filter applies to both sender (MM2S)
// and receiver (S2MM) BDs.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu1_1col' | FileCheck %s

// Two packet flows (pkt_a + pkt_b) both land at tile(0,2) S2MM 0 because
// TileDMAAllocator collapses them onto one physical L1 channel. Each
// receiver BD now carries the right packet-info filter.
// CHECK: aie.tile(0, 2)
// CHECK: aie.mem(%[[tile_0_2:.*]])
// CHECK: aie.dma_start(S2MM, 0
// CHECK-DAG: aie.dma_bd({{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = [[pkt_a:[0-9]+]]>
// CHECK-DAG: aie.dma_bd({{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = [[pkt_b:[0-9]+]]>

module {
  air.channel @pkt_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkt_b [1, 1] {channel_type = "npu_dma_packet"}
  func.func @func_recv_demux(%a: memref<64xi32>, %b: memref<64xi32>) {
    air.channel.put @pkt_a[] (%a[] [] []) {id = 1 : i32} : (memref<64xi32>)
    air.channel.put @pkt_b[] (%b[] [] []) {id = 2 : i32} : (memref<64xi32>)
    air.segment @seg attributes {x_loc = 0 : i64, x_size = 1 : i64,
                                  y_loc = 2 : i64, y_size = 1 : i64} {
      %c1 = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx = %c1, %sy = %c1)
          attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %l1_a = memref.alloc() : memref<64xi32, 2>
        %l1_b = memref.alloc() : memref<64xi32, 2>
        air.channel.get @pkt_a[%tx, %ty] (%l1_a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
        air.channel.get @pkt_b[%tx, %ty] (%l1_b[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        memref.dealloc %l1_a : memref<64xi32, 2>
        memref.dealloc %l1_b : memref<64xi32, 2>
      }
    }
    return
  }
}
