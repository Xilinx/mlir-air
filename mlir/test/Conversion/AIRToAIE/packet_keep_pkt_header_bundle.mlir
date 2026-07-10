//===- packet_keep_pkt_header_bundle.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// keep_pkt_header is per-FLOW when a bundle is split: two producers gather into
// one L2 buffer, and only the split whose GET lands at offset 0 keeps the
// routing header; the offset-8 split strips it. Both producer BDs stay
// unstamped (air.src_writes_pkt_header marks the whole bundle).

// No BD (producer or receiver) is statically stamped -- the kernel wrote the
// header itself.
// CHECK-NOT: packet = #aie.packet_info

// Bundle split: offset-0 bearer keeps keep_pkt_header; sibling only marks
// src_writes_pkt_header (so its producer BD is still unstamped, but the flow
// strips the header).
// CHECK: air.channel @channel_0 [1, 1] {air.src_writes_pkt_header, channel_type = "npu_dma_packet", keep_pkt_header}
// CHECK: air.channel @channel_1 [1, 1] {air.src_writes_pkt_header, channel_type = "npu_dma_packet"}

// Exactly one flow keeps the header at the switchbox; the sibling does not.
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK-NEXT: } {keep_pkt_header = true}
// The sibling flow's closing brace carries no attribute dictionary.
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, DMA : 1>
// CHECK-NEXT: }

air.channel @k [2, 1] {channel_type = "npu_dma_packet", keep_pkt_header}
func.func @f() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<2x8xbf16, 1>) {
        %alloc = memref.alloc() : memref<2x8xbf16, 1>
        air.execute_terminator %alloc : memref<2x8xbf16, 1>
      }
      air.channel.get @k[%c0,   %c0] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      air.channel.get @k[%c1_0, %c0] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<2x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<2x8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        %z0 = arith.constant 0 : index
        air.channel.put @k[%z0, %z0] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0) attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        %z1 = arith.constant 1 : index
        %z0 = arith.constant 0 : index
        air.channel.put @k[%z1, %z0] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
