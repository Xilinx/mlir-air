//===- consumerless_memtile_drain.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// An L2 memtile buffer that is DMA-filled (a channel.get writes into it) but is
// never read out (no channel.put reads it) within the segment is a "pure drain"
// -- its data is discarded. Its receiving S2MM BD must SELF-RECYCLE on a single
// lock (acquire AND release the same lock) so it re-arms every dispatch. The
// legacy producer/consumer lock pair would fire once, then deadlock on the next
// dispatch re-acquiring the producer lock (no consumer ever releases it).
//
// Here @live_in feeds an L2 buffer that IS forwarded to a core (a normal
// relay, keeping a distinct producer/consumer lock pair), while @drain_in feeds
// a consumerless L2 buffer (memref<32xi32, 1>) that gets the self lock.

// CHECK: aie.memtile_dma
// The live relay buffer (memref<64xi32, 1>) appears as two BDs (MM2S drain to
// the core + S2MM fill); advance past both.
// CHECK: aie.dma_bd(%{{.*}} : memref<64xi32, 1>
// CHECK: aie.dma_bd(%{{.*}} : memref<64xi32, 1>
// The consumerless drain's S2MM BD (memref<32xi32, 1>) acquires and releases
// the SAME lock (self-recycling), not a distinct producer/consumer pair.
// CHECK: aie.use_lock(%[[DRAIN:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<32xi32, 1>
// CHECK-NEXT: aie.use_lock(%[[DRAIN]], Release, %{{.*}})

module {
  air.channel @live_in [1, 1]
  air.channel @drain_in [1, 1]
  air.channel @to_core [1, 1]
  func.func @func_drain(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.channel.put @live_in[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
    air.channel.put @drain_in[] (%arg1[] [] []) {id = 2 : i32} : (memref<32xi32>)
    air.segment @seg0 {
      %c1_0 = arith.constant 1 : index
      %l2_live = memref.alloc() : memref<64xi32, 1>
      %l2_drain = memref.alloc() : memref<32xi32, 1>
      air.channel.get @live_in[] (%l2_live[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      air.channel.get @drain_in[] (%l2_drain[] [] []) {id = 4 : i32} : (memref<32xi32, 1>)
      air.channel.put @to_core[] (%l2_live[] [] []) {id = 5 : i32} : (memref<64xi32, 1>)
      air.herd tile(%tx, %ty) in (%sx = %c1_0, %sy = %c1_0) attributes {sym_name = "herd0"} {
        %buf = memref.alloc() : memref<64xi32, 2>
        air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 6 : i32} : (memref<64xi32, 2>)
        memref.dealloc %buf : memref<64xi32, 2>
      }
      memref.dealloc %l2_live : memref<64xi32, 1>
      memref.dealloc %l2_drain : memref<32xi32, 1>
    }
    return
  }
}
