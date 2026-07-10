//===- packet_single_id_bd_tag.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// A channel pinning a SINGLE packet_id emits one packet_flow with that id AND
// stamps the producer DMA BD with it (recorded in packetIDForChannelName ->
// generateDmaBd). This is the DMA-stamped path (contrast the multi-id
// kernel-header contract, where BDs are deliberately left untagged).

// CHECK: aie.dma_bd(%{{.*}} : memref<80xbf16, 2>, 14, 66) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>
// CHECK: aie.packet_flow(3) {

air.channel @m [1, 1] {channel_type = "npu_dma_packet", packet_ids = [3]}
func.func @single_id() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %t, %l2 = air.execute -> (memref<66xbf16, 1>) {
        %alloc = memref.alloc() : memref<66xbf16, 1>
        air.execute_terminator %alloc : memref<66xbf16, 1>
      }
      air.channel.get @m[] (%l2[] [] []) : (memref<66xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<66xbf16, 1> }
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
