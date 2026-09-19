//===- memtile_no_chain_lock_fanin_ignored.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Scope test: the `air.no_chain_lock` opt-out is honored ONLY for fan-out. A
// fan-in buffer (4 sub-region writers + 1 full reader) still gets the v2
// rendezvous locks even when tagged, because reverting fan-in to the legacy
// counted lock would reintroduce the write-side race the v2 template exists
// to prevent. So this tagged fan-in must lower to per-slot (cap, sig) pairs
// plus a ping-pong twin.
//
// Note what the tag was FOR: over-serialization of independent participants.
// The per-slot rendezvous no longer serializes them, so on the fan-out side
// the opt-out is now a no-op in practice; it stays honored there so designs
// carrying it keep their existing lowering.

// CHECK: aie.device
// CHECK-DAG: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// v2 locks emitted despite the tag: two capacity locks primed to the
// participant count (4) and two signal locks at 0, one pair per slot.
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 4 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 4 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}

// Two buffer instances (primary + ping-pong twin): the chain-lock path ran.
// CHECK-DAG: aie.buffer(%[[MT]]) {{.*}} : memref<4x8xbf16, 1
// CHECK-DAG: aie.buffer(%[[MT]]) {{.*}} : memref<4x8xbf16, 1

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @w3 [1, 1]
air.channel @r0 [1, 1]
func.func @memtile_fanin_no_chain_lock_ignored() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      // Fan-in buffer tagged air.no_chain_lock: the tag is IGNORED for fan-in.
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split, air.no_chain_lock} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w3[] (%l2[%c3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
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
