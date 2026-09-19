//===- memtile_chain_lock_v2_fanout.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// v2 rendezvous-lock test: shared L2 buffer with 1 full writer + 4 sub-region
// readers (fan-out), with 2-slot ping-pong. Mirror image of the fan-in test:
//   - one (cap, sig) lock pair PER BUFFER SLOT: 2 locks at init=4 (the
//     participant count) and 2 at init=0
//   - TWO aie.buffer instances of the same memref type (primary + twin)
//   - the writer, on slot s, acquires cap[s] by 4 and releases sig[s] by 4
//   - every reader, on slot s, acquires sig[s] by 1 and releases cap[s] by 1
//   - each channel's BD chain alternates between primary and twin buffers
//
// As in the fan-in case, the load-bearing property is that ALL FOUR readers
// acquire the SAME lock for a given slot, so no reader waits on another
// reader. The predecessor chained them (writer -> R0 -> R1 -> R2 -> R3 ->
// cap), which over-serializes independent consumers and can deadlock against
// switchbox arbiter sharing -- see the fan-in test for the full mechanism.

// CHECK: aie.device
// CHECK-DAG: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// Per-slot pairs: two capacity locks primed to the participant count (4) and
// two signal locks at 0 (the predecessor had ONE cap at init=2 instead, with
// the slot count encoded in its init). Every lock is bound by name in the
// memtile_dma checks below, which pin the structure exactly.
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 4 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 4 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}

// Two shared buffer instances (primary + ping-pong twin) of matching type.
// CHECK-DAG: aie.buffer(%[[MT]]) {{.*}} : memref<4x8xbf16, 1
// CHECK-DAG: aie.buffer(%[[MT]]) {{.*}} : memref<4x8xbf16, 1

// CHECK: aie.memtile_dma(%[[MT]])

// Reader 0 binds the slot locks: one credit each way, on its own slice.
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[SIG0:.*]], AcquireGreaterEqual, %[[C1:.*]])
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.use_lock(%[[CAP0:.*]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1:.*]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.use_lock(%[[CAP1:.*]], Release, %[[C1]])

// Readers 1-3: THE SAME slot locks as reader 0 -- mutually unordered.
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.use_lock(%[[SIG0]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[CAP0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], Release, %[[C1]])
// CHECK: aie.dma_start(MM2S, 2
// CHECK: aie.use_lock(%[[SIG0]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[CAP0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], Release, %[[C1]])
// CHECK: aie.dma_start(MM2S, 3
// CHECK: aie.use_lock(%[[SIG0]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[CAP0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], Release, %[[C1]])

// The single writer fills a whole slot: acquire capacity by N=4 (all four
// readers done with it) and release N read credits.
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[CAP0]], AcquireGreaterEqual, %[[C4:.*]])
// CHECK: aie.dma_bd({{.*}} : memref<4x8xbf16, 1> offset = 0 len = 32)
// CHECK: aie.use_lock(%[[SIG0]], Release, %[[C4]])
// CHECK: aie.use_lock(%[[CAP1]], AcquireGreaterEqual, %[[C4]])
// CHECK: aie.dma_bd({{.*}} : memref<4x8xbf16, 1> offset = 0 len = 32)
// CHECK: aie.use_lock(%[[SIG1]], Release, %[[C4]])

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
air.channel @r3 [1, 1]
func.func @memtile_fanout_chain_lock_v2() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      // Shared L2 buffer (air.no_split): 1 full write + 4 sub-region reads.
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      // 1 full-buffer write (from producer herd via @w0)
      air.channel.get @w0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      // 4 sub-region reads to 4 consumer herds via @r0..@r3
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r3[] (%l2[%c3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
      // 1 producer herd
      air.herd @hw tile (%txw, %tyw) in (%sxw=%c1_0, %syw=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dw = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
      // 4 consumer herds, each reads 8 bf16
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx1, %ty1) in (%sx1=%c1_0, %sy1=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d1 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx2, %ty2) in (%sx2=%c1_0, %sy2=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d2 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h3 tile (%tx3, %ty3) in (%sx3=%c1_0, %sy3=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d3 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
