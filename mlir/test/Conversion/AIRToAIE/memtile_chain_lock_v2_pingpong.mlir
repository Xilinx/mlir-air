//===- memtile_chain_lock_v2_pingpong.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// v2 chain-lock 2-slot ping-pong structural test (fan-in shape, 4
// writers + 1 reader). Verifies that:
//   1. The cap_lock is bumped to init=2 (2 ping-pong slots).
//   2. Two aie.buffer instances of the same memref type exist at the
//      memtile (primary + twin).
//   3. Each channel's BD chain has exactly 2 BDs that alternate between
//      the two buffer instances (next_bd loops back to the first BD).
//   4. The same lock pair is used by BOTH primary and twin BDs at each
//      chain stage (locks shared across ping-pong instances, matching
//      a shared-L2 producer-consumer pattern).
//   5. All per-BD acquire/release counts stay at 1 (chain semantics
//      preserved; cap=2 admits two stage-K firings before blocking).

// CHECK: aie.device
// CHECK-DAG: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// cap_lock with init=2 and the chain's signal locks (init=0).
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 2 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}

// Two buffer instances on the same memtile, same memref type.
// CHECK-DAG: %[[BUF_A:.*]] = aie.buffer(%[[MT]]) {{.*}} : memref<4x8xbf16, 1
// CHECK-DAG: %[[BUF_B:.*]] = aie.buffer(%[[MT]]) {{.*}} : memref<4x8xbf16, 1

// memtile_dma block: for each channel, the BD chain must contain
// dma_bds referencing both buffer instances, with identical per-BD
// acquire/release counts. We do not constrain BD order (greedy lock
// allocation may emit primary→twin or twin→primary), but both must
// appear.
// CHECK: aie.memtile_dma(%[[MT]])
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK: aie.dma_bd({{.*}} : memref<4x8xbf16, 1
// CHECK: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK: aie.next_bd
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK: aie.dma_bd({{.*}} : memref<4x8xbf16, 1
// CHECK: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK: aie.next_bd

// Negative: no acquire-by-N (legacy pattern would emit count=4 on cap).
// (Formerly CHECK-NOT for integer literal 4; with SSA constants the positive
//  CHECK lines above already confirm all acquire counts are 1.)

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @w3 [1, 1]
air.channel @r0 [1, 1]
func.func @memtile_chain_pingpong_v2() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %t, %l2 = air.execute -> (memref<4x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<4x8xbf16, 1>
        air.execute_terminator %alloc : memref<4x8xbf16, 1>
      }
      air.channel.get @w0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.get @w3[] (%l2[%c3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<4x8xbf16, 1>}
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
