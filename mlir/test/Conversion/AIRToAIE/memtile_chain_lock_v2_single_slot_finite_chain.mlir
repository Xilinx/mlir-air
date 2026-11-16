//===- memtile_chain_lock_v2_single_slot_finite_chain.mlir -----*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// A chain-locked memtile buffer that each of its channels moves once, in a
// task of its own, while one of those channels then carries another buffer
// in a loop. That channel's chain is finite and runs each BD once, so the
// buffer gets no ping-pong twin: a twin BD would wait for a fill that never
// comes.

// Fan-in: four writers, one reader whose channel then carries the loop.

// CHECK-LABEL: aie.device
// CHECK: aie.buffer({{.*}}) {{.*}} : memref<4x8xbf16, 1
// CHECK-NOT: aie.buffer({{.*}}) {{.*}} : memref<4x8xbf16, 1
// CHECK: aie.memtile_dma

air.channel @w0 [1, 1]
air.channel @w1 [1, 1]
air.channel @w2 [1, 1]
air.channel @w3 [1, 1]
air.channel @w4 [1, 1]
air.channel @r0 [1, 1]
func.func @memtile_chain_single_slot_finite_chain() {
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
      scf.for %i = %c0 to %c2 step %c1_0 {
        %tb, %l2b = air.execute -> (memref<8xbf16, 1>) {
          %allocb = memref.alloc() {air.no_split} : memref<8xbf16, 1>
          air.execute_terminator %allocb : memref<8xbf16, 1>
        }
        air.channel.get @w4[] (%l2b[] [] []) : (memref<8xbf16, 1>)
        air.channel.put @r0[] (%l2b[] [] []) : (memref<8xbf16, 1>)
        %db = air.execute {memref.dealloc %l2b : memref<8xbf16, 1>}
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
        %c0_r = arith.constant 0 : index
        %c1_r = arith.constant 1 : index
        %c2_r = arith.constant 2 : index
        scf.for %j = %c0_r to %c2_r step %c1_r {
          %tr2, %l1b = air.execute -> (memref<8xbf16, 2>) {
            %bb = memref.alloc() : memref<8xbf16, 2>
            air.execute_terminator %bb : memref<8xbf16, 2>
          }
          air.channel.get @r0[] (%l1b[] [] []) : (memref<8xbf16, 2>)
          %dr2 = air.execute {memref.dealloc %l1b : memref<8xbf16, 2>}
        }
      }
      air.herd @h4 tile (%tx4, %ty4) in (%sx4=%c1_0, %sy4=%c1_0)
            attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
        %c0_w = arith.constant 0 : index
        %c1_w = arith.constant 1 : index
        %c2_w = arith.constant 2 : index
        scf.for %k = %c0_w to %c2_w step %c1_w {
          %tw, %l1w = air.execute -> (memref<8xbf16, 2>) {
            %ww = memref.alloc() : memref<8xbf16, 2>
            air.execute_terminator %ww : memref<8xbf16, 2>
          }
          air.channel.put @w4[] (%l1w[] [] []) : (memref<8xbf16, 2>)
          %dw = air.execute {memref.dealloc %l1w : memref<8xbf16, 2>}
        }
      }
    }
  }
  return
}

// -----

// Fan-out: one writer whose channel then carries the loop, four readers.

// CHECK-LABEL: aie.device
// CHECK: aie.buffer({{.*}}) {{.*}} : memref<4x8xbf16, 1
// CHECK-NOT: aie.buffer({{.*}}) {{.*}} : memref<4x8xbf16, 1
// CHECK: aie.memtile_dma

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
air.channel @r3 [1, 1]
air.channel @r4 [1, 1]
func.func @fanout_writer_finite_chain() {
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
      air.channel.get @w0[] (%l2[] [] []) : (memref<4x8xbf16, 1>)
      air.channel.put @r0[] (%l2[%c0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r1[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r2[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      air.channel.put @r3[] (%l2[%c3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<4x8xbf16, 1>)
      %d_ = air.execute {memref.dealloc %l2 : memref<4x8xbf16, 1>}
      scf.for %i = %c0 to %c2 step %c1_0 {
        %tb, %l2b = air.execute -> (memref<8xbf16, 1>) {
          %allocb = memref.alloc() {air.no_split} : memref<8xbf16, 1>
          air.execute_terminator %allocb : memref<8xbf16, 1>
        }
        air.channel.get @w0[] (%l2b[] [] []) : (memref<8xbf16, 1>)
        air.channel.put @r4[] (%l2b[] [] []) : (memref<8xbf16, 1>)
        %db = air.execute {memref.dealloc %l2b : memref<8xbf16, 1>}
      }
      air.herd @hw tile (%txw, %tyw) in (%sxw=%c1_0, %syw=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dw = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
        %c0_w = arith.constant 0 : index
        %c1_w = arith.constant 1 : index
        %c2_w = arith.constant 2 : index
        scf.for %k = %c0_w to %c2_w step %c1_w {
          %tw, %l1w = air.execute -> (memref<8xbf16, 2>) {
            %ww = memref.alloc() : memref<8xbf16, 2>
            air.execute_terminator %ww : memref<8xbf16, 2>
          }
          air.channel.put @w0[] (%l1w[] [] []) : (memref<8xbf16, 2>)
          %dw2 = air.execute {memref.dealloc %l1w : memref<8xbf16, 2>}
        }
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
      air.herd @h4 tile (%tx4, %ty4) in (%sx4=%c1_0, %sy4=%c1_0)
            attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
        %c0_r = arith.constant 0 : index
        %c1_r = arith.constant 1 : index
        %c2_r = arith.constant 2 : index
        scf.for %j = %c0_r to %c2_r step %c1_r {
          %tr, %l1r = air.execute -> (memref<8xbf16, 2>) {
            %bb = memref.alloc() : memref<8xbf16, 2>
            air.execute_terminator %bb : memref<8xbf16, 2>
          }
          air.channel.get @r4[] (%l1r[] [] []) : (memref<8xbf16, 2>)
          %dr = air.execute {memref.dealloc %l1r : memref<8xbf16, 2>}
        }
      }
    }
  }
  return
}
