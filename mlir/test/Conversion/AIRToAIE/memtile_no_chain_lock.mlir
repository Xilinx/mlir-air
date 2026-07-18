//===- memtile_no_chain_lock.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Opt-out test: the same 1-writer + 4-reader fan-out shared L2 buffer that the
// v2 chain-lock lowers to a daisy-chain (1 cap init=2 + 4 init=0 signal locks +
// a ping-pong twin buffer) instead keeps the LEGACY counted-lock template when
// its source alloc is pinned with `air.no_chain_lock` -- even with the v2 fix
// requested. This is for fan-out broadcast buffers whose N readers are
// independent compute cores: daisy-chaining them serializes independent readers
// unnecessarily and can deadlock when competing with a fan-in chain.

// CHECK: aie.device
// CHECK-DAG: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// Legacy counted-lock cap (init = number of readers = 4), NOT the v2 chain-lock
// cap (init = 2) and NOT the 4 per-reader init=0 signal locks.
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 4 : i32}
// CHECK-NOT: aie.lock(%[[MT]], {{[0-9]+}}) {init = 2 : i32}

// Exactly ONE buffer instance (no ping-pong twin), carrying the opt-out pin.
// CHECK: aie.buffer(%[[MT]]) {{.*}}air.no_chain_lock{{.*}} : memref<4x8xbf16, 1
// CHECK-NOT: aie.buffer(%[[MT]])

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
air.channel @r3 [1, 1]
func.func @memtile_fanout_no_chain_lock() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      // Shared L2 buffer pinned air.no_chain_lock: 1 full write + 4 sub-region reads.
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split, air.no_chain_lock} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r3[] (%l2[%c3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {
        memref.dealloc %l2 : memref<4x8xbf16, 1>
      }
      air.herd @hw tile (%txw, %tyw) in (%sxw=%c1_0, %syw=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dw = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
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
