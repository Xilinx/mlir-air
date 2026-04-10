//===- shim_pkt_channel_sharing.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that multiple dma_packet channels from L3 to L2 on the same shim column
// can share physical DMA channels via packet-flow time multiplexing, exceeding
// the 2 physical MM2S channel limit per shim tile.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu1_1col' | FileCheck %s

// Three packet-flow L3-to-L2 input channels sharing one shim tile (col 0).
// With 2 physical MM2S channels, the third must share via packet-flow reuse.
// All three should get packet_flow ops with unique IDs and shim_dma_allocation
// ops sharing the same MM2S channel.
// CHECK: aie.packet_flow(0)
// CHECK: aie.packet_flow(1)
// CHECK: aie.packet_flow(2)
// CHECK: aie.shim_dma_allocation @air_out({{.*}}, S2MM, 0)
// CHECK: aie.shim_dma_allocation @air_pkt_in_0({{.*}}, MM2S, 0)
// CHECK: aie.shim_dma_allocation @air_pkt_in_1({{.*}}, MM2S, 0)
// CHECK: aie.shim_dma_allocation @air_pkt_in_2({{.*}}, MM2S, 0)

module {
  air.channel @pkt_in_0 [1, 1] {channel_type = "dma_packet"}
  air.channel @pkt_in_1 [1, 1] {channel_type = "dma_packet"}
  air.channel @pkt_in_2 [1, 1] {channel_type = "dma_packet"}
  air.channel @to_core [1, 1]
  air.channel @from_core [1, 1]
  air.channel @out [1, 1]
  func.func @func_pkt_sharing(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    air.channel.put @pkt_in_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
    air.channel.put @pkt_in_1[] (%arg1[] [] []) {id = 2 : i32} : (memref<64xi32>)
    air.channel.put @pkt_in_2[] (%arg2[] [] []) {id = 3 : i32} : (memref<64xi32>)
    air.segment @seg0 {
      %c1_0 = arith.constant 1 : index
      %l2_0 = memref.alloc() : memref<64xi32, 1>
      %l2_1 = memref.alloc() : memref<64xi32, 1>
      %l2_2 = memref.alloc() : memref<64xi32, 1>
      air.channel.get @pkt_in_0[] (%l2_0[] [] []) {id = 4 : i32} : (memref<64xi32, 1>)
      air.channel.get @pkt_in_1[] (%l2_1[] [] []) {id = 5 : i32} : (memref<64xi32, 1>)
      air.channel.get @pkt_in_2[] (%l2_2[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
      air.channel.put @to_core[] (%l2_0[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
      air.herd tile(%tx, %ty) in (%sx = %c1_0, %sy = %c1_0) attributes {sym_name = "herd0"} {
        %buf = memref.alloc() : memref<64xi32, 2>
        %res = memref.alloc() : memref<64xi32, 2>
        air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
        air.channel.put @from_core[%tx, %ty] (%res[] [] []) {id = 9 : i32} : (memref<64xi32, 2>)
        memref.dealloc %buf : memref<64xi32, 2>
        memref.dealloc %res : memref<64xi32, 2>
      }
      %l2_out = memref.alloc() : memref<64xi32, 1>
      air.channel.get @from_core[] (%l2_out[] [] []) {id = 10 : i32} : (memref<64xi32, 1>)
      air.channel.put @out[] (%l2_out[] [] []) {id = 11 : i32} : (memref<64xi32, 1>)
      memref.dealloc %l2_0 : memref<64xi32, 1>
      memref.dealloc %l2_1 : memref<64xi32, 1>
      memref.dealloc %l2_2 : memref<64xi32, 1>
      memref.dealloc %l2_out : memref<64xi32, 1>
    }
    air.channel.get @out[] (%arg3[] [] []) {id = 12 : i32} : (memref<64xi32>)
    return
  }
}
