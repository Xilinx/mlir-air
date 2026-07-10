//===- packet_multi_id_no_bd_tag.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Kernel-header contract: a channel pinning MULTIPLE packet_ids emits one
// packet_flow per id but DELIBERATELY leaves the DMA BDs untagged -- the compute
// core writes the routing id into the payload header, so the DMA must not stamp
// a packet header (packetIDForChannelName is intentionally left unset for >1
// pinned id). The producer BD carries only its task_id, no packet_info.

// CHECK: aie.dma_bd(%{{.*}} : memref<80xbf16, 2>, 14, 66) {task_id = 0 : i32}
// CHECK-NOT: packet = #aie.packet_info
// CHECK: aie.packet_flow(2) {
// CHECK: aie.packet_flow(5) {

air.channel @m [1, 1] {channel_type = "npu_dma_packet", packet_ids = [2, 5]}
func.func @multi_id_no_bd_tag() {
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
