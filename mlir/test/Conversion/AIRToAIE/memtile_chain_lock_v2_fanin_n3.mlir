//===- memtile_chain_lock_v2_fanin_n3.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// The rendezvous credit count is the PARTICIPANT count, not a constant. Same
// fan-in shape as memtile_chain_lock_v2_fanin.mlir with three writers instead
// of four, so the capacity locks prime to 3 and the reader acquires/releases
// 3. Three also exercises an odd participant count, where the slot count (2)
// does not divide the credit count.

// CHECK: aie.device
// CHECK-DAG: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// One (cap, sig) pair per slot: capacity primed to 3, signal to 0.
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 3 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 3 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}

// CHECK: aie.memtile_dma(%[[MT]])

// Reader: acquires a slot's signal lock by 3 and hands 3 capacity credits
// back. Binds the slot locks for the writer checks.
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[SIG0:.*]], AcquireGreaterEqual, %[[C3:.*]])
// CHECK: aie.dma_bd({{.*}} : memref<3x8xbf16, 1> offset = 0 len = 24)
// CHECK: aie.use_lock(%[[CAP0:.*]], Release, %[[C3]])
// CHECK: aie.use_lock(%[[SIG1:.*]], AcquireGreaterEqual, %[[C3]])
// CHECK: aie.dma_bd({{.*}} : memref<3x8xbf16, 1> offset = 0 len = 24)
// CHECK: aie.use_lock(%[[CAP1:.*]], Release, %[[C3]])

// All three writers share each slot's pair -- mutually unordered.
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[CAP0]], AcquireGreaterEqual, %[[C1:.*]])
// CHECK: aie.use_lock(%[[SIG0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], Release, %[[C1]])
// CHECK: aie.dma_start(S2MM, 1
// CHECK: aie.use_lock(%[[CAP0]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], Release, %[[C1]])
// CHECK: aie.dma_start(S2MM, 2
// CHECK: aie.use_lock(%[[CAP0]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], Release, %[[C1]])

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @r0 [1, 1]
func.func @memtile_fanin_n3_chain_lock_v2() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<3x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<3x8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d2 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<24xbf16, 2>) {
          %aa = memref.alloc() : memref<24xbf16, 2>
          air.execute_terminator %aa : memref<24xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<24xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<24xbf16, 2>}
      }
    }
  }
  return
}
