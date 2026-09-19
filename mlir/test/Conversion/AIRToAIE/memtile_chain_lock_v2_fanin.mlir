//===- memtile_chain_lock_v2_fanin.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// v2 rendezvous-lock test: shared L2 buffer with 4 sub-region writers + 1 full
// reader (fan-in), with 2-slot ping-pong. Expected:
//   - one (cap, sig) lock pair PER BUFFER SLOT: 2 locks at init=4 (the
//     participant count) and 2 at init=0
//   - TWO aie.buffer instances of the same memref type (primary + twin)
//   - every writer, on slot s, acquires cap[s] by 1 and releases sig[s] by 1
//   - the reader, on slot s, acquires sig[s] by 4 and releases cap[s] by 4
//   - each channel's BD chain alternates between primary and twin buffers
//
// The load-bearing property is that ALL FOUR writers acquire the SAME lock for
// a given slot. Writers are mutually unordered, so a writer that arrives early
// never waits on another writer.
//
// That is a fix, not a detail. The predecessor daisy-chained the writers
// (cap -> W0 -> W1 -> W2 -> W3 -> R -> cap), imposing a compile-time total
// order on arrivals that are independent at runtime. When the pathfinder packs
// two of those streams onto one switchbox arbiter -- which it must, a shim
// switchbox having 6 arbiters and a busy column more masters than that -- an
// early-arriving LATE writer stalls on its chain predecessor while holding the
// arbiter the EARLY writer needs, and the chain never advances. Measured on
// gemma4-e2b: ping-pong let the two east projection columns (stages 2 and 3)
// run a round ahead, stages 0 and 2 shared arbiter 1 in shim tile (2,0), and
// decode hung in firmware TDR.

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

// The single reader drains a whole slot: acquire its signal lock by N=4 (one
// credit from each writer) and hand N capacity credits back. Binds the slot
// locks for the writer checks below.
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[SIG0:.*]], AcquireGreaterEqual, %[[C4:.*]])
// CHECK: aie.dma_bd({{.*}} : memref<4x8xbf16, 1> offset = 0 len = 32)
// CHECK: aie.use_lock(%[[CAP0:.*]], Release, %[[C4]])
// CHECK: aie.use_lock(%[[SIG1:.*]], AcquireGreaterEqual, %[[C4]])
// CHECK: aie.dma_bd({{.*}} : memref<4x8xbf16, 1> offset = 0 len = 32)
// CHECK: aie.use_lock(%[[CAP1:.*]], Release, %[[C4]])

// Writer 0: slot 0 then slot 1, one credit each way.
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[CAP0]], AcquireGreaterEqual, %[[C1:.*]])
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.use_lock(%[[SIG0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.use_lock(%[[SIG1]], Release, %[[C1]])

// Writers 1-3: THE SAME slot locks as writer 0. This is the anti-deadlock
// invariant -- under the old daisy chain these would have been sig[0], sig[1]
// and sig[2], each writer gated on its predecessor.
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
// CHECK: aie.dma_start(S2MM, 3
// CHECK: aie.use_lock(%[[CAP0]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG0]], Release, %[[C1]])
// CHECK: aie.use_lock(%[[CAP1]], AcquireGreaterEqual, %[[C1]])
// CHECK: aie.use_lock(%[[SIG1]], Release, %[[C1]])

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @w3 [1, 1]
air.channel @r0 [1, 1]
func.func @memtile_fanin_chain_lock_v2() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      // Shared L2 buffer carrying air.no_split.
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      // 4 segment-side gets from 4 herds — disjoint sub-regions of L2.
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w3[] (%l2[%c3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      // 1 full-buffer put (segment-side reader → goes to a consumer herd).
      air.channel.put @r0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
      // 4 producer herds — each pushes one 8-bf16 chunk into @w_i.
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
      air.herd @h3 tile (%tx3, %ty3) in (%sx3=%c1_0, %sy3=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @w3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d3 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      // 1 consumer herd reads the full assembled 32-bf16 buffer.
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
    }
  }
  return
}
