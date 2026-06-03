//===- shim_pkt_l2_l1_col_bucket_merge.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: ShimDMAAllocator must not split L3->memtile and L3->core
// flows into separate shim buckets when they target the same physical
// column.
//
// Before the fix, the bucket key was (col when known, else far-side
// Operation*). A flow whose far side was an unhinted memtile LTO (col=-1
// under Path B's saturation-fallback) keyed by Op*; a flow whose far side
// was a placed core kept keying by col. The two key universes never met,
// so a memtile-routed packet channel and a direct-to-core packet channel
// sharing the same column ended up on different shim LTOs, breaking the
// packet-multiplex branch and consuming twice as many physical shim NOC
// tiles. Multi-launch designs (e.g. la_lgu_ld_cascade_fused) hit the
// NPU2 8-shim budget as a result.
//
// The fix: when col is unknown but the far side is a memtile LTO, derive
// the bucket col by walking to the memtile's downstream cores (which
// carry concrete x_loc). The memtile and the direct-to-core flows both
// land in bucket col=0 and share ONE shim LTO via the existing
// packet-flow multiplex logic.
//
// Saturation is the trigger: two L2 memref buckets (operand classes A and
// B) both targeting col=0 makes L2MemrefToMemTileMap fall back to the
// round-robin path that leaves memtile LTOs unhinted (col=nullptr).
// Without that, single-bucket designs hit the col-affinity path which
// stamps col=0 on the memtile and masks the bug.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu1' | FileCheck %s

// CHECK-LABEL: aie.device(npu1) @seg
// All three packet channels share ONE shim LTO at MM2S 0 (packet
// time-multiplex). Pre-fix the direct-to-core channel got its own shim
// LTO because its bucket key (col=0) didn't match the memtile-routed
// channels' bucket key (Op*=memtile LTO).
// CHECK:        %[[shim:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK-NOT:    aie.logical_tile<ShimNOCTile>
// CHECK-DAG:    aie.shim_dma_allocation @air_pkt_a(%[[shim]], MM2S, 0)
// CHECK-DAG:    aie.shim_dma_allocation @air_pkt_b(%[[shim]], MM2S, 0)
// CHECK-DAG:    aie.shim_dma_allocation @air_pkt_c(%[[shim]], MM2S, 0)

module {
  // Two L3->L2 packet channels feeding col 0 via memtiles (operand
  // classes A and B). Two buckets at col 0 trigger the saturation
  // fallback in L2MemrefToMemTileMap, leaving memtiles col-unhinted.
  air.channel @pkt_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkt_b [1, 1] {channel_type = "npu_dma_packet"}
  // One L3->L1 direct packet channel feeding col 0.
  air.channel @pkt_c [1, 1] {channel_type = "npu_dma_packet"}
  // L2->L1 legs anchoring the memtile-routed flows to col 0 cores.
  air.channel @a_l2_to_core [1, 1]
  air.channel @b_l2_to_core [1, 1]

  func.func @func_shared_col(%a: memref<64xi32>, %b: memref<64xi32>,
                              %c: memref<64xi32>) {
    air.channel.put @pkt_a[] (%a[] [] []) {id = 1 : i32} : (memref<64xi32>)
    air.channel.put @pkt_b[] (%b[] [] []) {id = 2 : i32} : (memref<64xi32>)
    air.channel.put @pkt_c[] (%c[] [] []) {id = 3 : i32} : (memref<64xi32>)
    air.segment @seg attributes {x_loc = 0 : i64, x_size = 4 : i64,
                                  y_loc = 2 : i64, y_size = 1 : i64} {
      %c1 = arith.constant 1 : index
      // Two distinct operand-class shapes (8 vs 16 elements) so the two
      // buckets stay disjoint in L2MemrefToMemTileMap.
      %l2_a = memref.alloc() : memref<8x8xi32, 1>
      %l2_b = memref.alloc() : memref<4x16xi32, 1>
      air.channel.get @pkt_a[] (%l2_a[] [] []) {id = 4 : i32} : (memref<8x8xi32, 1>)
      air.channel.get @pkt_b[] (%l2_b[] [] []) {id = 5 : i32} : (memref<4x16xi32, 1>)
      air.channel.put @a_l2_to_core[] (%l2_a[] [] []) {id = 6 : i32} : (memref<8x8xi32, 1>)
      air.channel.put @b_l2_to_core[] (%l2_b[] [] []) {id = 7 : i32} : (memref<4x16xi32, 1>)
      air.herd @h tile(%tx, %ty) in (%sx = %c1, %sy = %c1)
          attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %l1_a = memref.alloc() : memref<8x8xi32, 2>
        %l1_b = memref.alloc() : memref<4x16xi32, 2>
        %l1_c = memref.alloc() : memref<64xi32, 2>
        air.channel.get @a_l2_to_core[%tx, %ty] (%l1_a[] [] []) {id = 8 : i32} : (memref<8x8xi32, 2>)
        air.channel.get @b_l2_to_core[%tx, %ty] (%l1_b[] [] []) {id = 9 : i32} : (memref<4x16xi32, 2>)
        air.channel.get @pkt_c[%tx, %ty] (%l1_c[] [] []) {id = 10 : i32} : (memref<64xi32, 2>)
        memref.dealloc %l1_a : memref<8x8xi32, 2>
        memref.dealloc %l1_b : memref<4x16xi32, 2>
        memref.dealloc %l1_c : memref<64xi32, 2>
      }
      memref.dealloc %l2_a : memref<8x8xi32, 1>
      memref.dealloc %l2_b : memref<4x16xi32, 1>
    }
    return
  }
}
