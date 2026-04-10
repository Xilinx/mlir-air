//===- shared_shim_channel_packet_ids.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test: when multiple dma_packet flows share a single shim DMA
// channel, each flow must get a distinct packet ID. Previously,
// labelMemcpyOpsWithPacketFlow used source-only lookup which returned the
// same (last-walked) packet flow for all flows on the channel, causing all
// channel.put ops to get the same packet ID.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// Two distinct packet flows should be created with different IDs.
// CHECK-LABEL: aie.device
// CHECK-DAG:   aie.packet_flow(0)
// CHECK-DAG:   aie.packet_flow(1)

// Each channel.put at L3 level should get a distinct packet attribute.
// CHECK-LABEL: func.func @test_shared_shim_packet_ids
// CHECK-DAG:   air.channel.put{{.*}}@chan_a{{.*}}packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
// CHECK-DAG:   air.channel.put{{.*}}@chan_b{{.*}}packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>

module {
  // Two dma_packet channels from L3 to L1, sharing the same shim column.
  air.channel @chan_a [1, 1] {channel_type = "dma_packet"}
  air.channel @chan_b [1, 1] {channel_type = "dma_packet"}

  func.func @test_shared_shim_packet_ids(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in0=%arg0, %in1=%arg1) : memref<64xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      // L3-to-device channel puts (shim MM2S direction)
      %put_a = air.channel.put async @chan_a[%c0, %c0] (%in0[] [] []) {id = 1 : i32} : (memref<64xbf16>)
      %put_b = air.channel.put async @chan_b[%c0, %c0] (%in1[] [] []) {id = 2 : i32} : (memref<64xbf16>)

      %seg = air.segment @seg async [%put_a, %put_b] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_seg = arith.constant 1 : index
        %herd = air.herd @herd async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          // L1 buffers
          %async_a, %buf_a = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %async_b, %buf_b = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          // L1 channel gets (compute tile S2MM direction)
          %get_a = air.channel.get async [%async_a] @chan_a[%hc0, %hc0] (%buf_a[] [] []) {id = 3 : i32} : (memref<64xbf16, 2>)
          %get_b = air.channel.get async [%async_b] @chan_b[%hc0, %hc0] (%buf_b[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)

          %dealloc_a = air.execute [%get_a] {
            memref.dealloc %buf_a : memref<64xbf16, 2>
          }
          %dealloc_b = air.execute [%get_b] {
            memref.dealloc %buf_b : memref<64xbf16, 2>
          }
        }
      }
    }
    return
  }
}
