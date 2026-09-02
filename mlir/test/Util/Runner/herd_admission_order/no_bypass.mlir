//===- no_bypass.mlir                                 -*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// The same four stages of 4, 1, 4 and 2 tiles, offered in the right order, and
// still deadlocked without the head-of-line rule.
//
// The one-tile second stage does twice the work here, so it is still holding
// its tile while the four-tile third stage is refused -- three free, four
// wanted. That is the window: the two-tile fourth stage fits in three and, if
// it is offered them, takes exactly the tiles the third stage was waiting for,
// and the fourth is waiting on the third. Refusing to offer past a herd that
// could not fit is what closes it.
//
// Stalls at 415 cycles with the head-of-line rule reverted, and runs with the
// tie-break reverted -- so unlike candidate_order.mlir this one pins the
// head-of-line rule alone.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: Latency (all-iterations mode): 0.824us

module {
  air.channel @c1 [1, 1] {broadcast_shape = [1, 1]}
  air.channel @c2 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @c3 [1, 1] {broadcast_shape = [1, 2]}
  func.func @test(%arg0: memref<64xi8>) {
    %c1v = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1v, %sy=%c1v) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %1 = air.segment async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c4_s = arith.constant 4 : index
        %tk0, %k0 = air.execute -> (memref<8x8xi8, 1>) {
          %ak0 = memref.alloc() : memref<8x8xi8, 1>
          air.execute_terminator %ak0 : memref<8x8xi8, 1>
        }
        %tk1, %k1 = air.execute -> (memref<8x8xi8, 1>) {
          %ak1 = memref.alloc() : memref<8x8xi8, 1>
          air.execute_terminator %ak1 : memref<8x8xi8, 1>
        }
        %tk2, %k2 = air.execute -> (memref<8x8xi8, 1>) {
          %ak2 = memref.alloc() : memref<8x8xi8, 1>
          air.execute_terminator %ak2 : memref<8x8xi8, 1>
        }
        %tk3, %k3 = air.execute -> (memref<8x8xi8, 1>) {
          %ak3 = memref.alloc() : memref<8x8xi8, 1>
          air.execute_terminator %ak3 : memref<8x8xi8, 1>
        }
        %h0 = air.herd @first async [%tk0]  tile (%x0, %y0) in (%sx0=%c1_s, %sy0=%c4_s) args(%ka0=%k0) : memref<8x8xi8, 1> attributes {id = 10 : i32} {
          %c0_h0 = arith.constant 0 : index
          %tw0_0, %w0_0 = air.execute -> (memref<8x8xi8, 2>) {
            %aw0_0 = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %aw0_0 : memref<8x8xi8, 2>
          }
          %tv0_0, %v0_0 = air.execute -> (memref<8xi8, 2>) {
            %av0_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %av0_0 : memref<8xi8, 2>
          }
          %to0_0, %o0_0 = air.execute -> (memref<8xi8, 2>) {
            %ao0_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ao0_0 : memref<8xi8, 2>
          }
          %tc0_0 = air.execute [%tw0_0, %tv0_0, %to0_0] {
            linalg.matvec {air.op_cost = "priced"} ins(%w0_0, %v0_0 : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o0_0 : memref<8xi8, 2>)
          }
        }
        %tb0, %b0 = air.execute -> (memref<8xi8, 1>) {
          %ab0 = memref.alloc() : memref<8xi8, 1>
          air.execute_terminator %ab0 : memref<8xi8, 1>
        }
        %p0 = air.channel.put async [%h0, %tb0] @c1[%c0_s, %c0_s] (%b0[] [] []) {id = 200 : i32} : (memref<8xi8, 1>)
        %h1 = air.herd @second async [%tk1]  tile (%x1, %y1) in (%sx1=%c1_s, %sy1=%c1_s) args(%ka1=%k1) : memref<8x8xi8, 1> attributes {id = 11 : i32} {
          %c0_h1 = arith.constant 0 : index
          %tg1, %g1 = air.execute -> (memref<8xi8, 2>) {
            %ag1 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ag1 : memref<8xi8, 2>
          }
          %r1 = air.channel.get async [%tg1] @c1[%c0_h1, %c0_h1] (%g1[] [] []) {id = 101 : i32} : (memref<8xi8, 2>)
          %tw1_0, %w1_0 = air.execute -> (memref<8x8xi8, 2>) {
            %aw1_0 = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %aw1_0 : memref<8x8xi8, 2>
          }
          %tv1_0, %v1_0 = air.execute -> (memref<8xi8, 2>) {
            %av1_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %av1_0 : memref<8xi8, 2>
          }
          %to1_0, %o1_0 = air.execute -> (memref<8xi8, 2>) {
            %ao1_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ao1_0 : memref<8xi8, 2>
          }
          %tc1_0 = air.execute [%tw1_0, %tv1_0, %to1_0] {
            linalg.matvec {air.op_cost = "priced"} ins(%w1_0, %v1_0 : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o1_0 : memref<8xi8, 2>)
          }
          %tw1_1, %w1_1 = air.execute -> (memref<8x8xi8, 2>) {
            %aw1_1 = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %aw1_1 : memref<8x8xi8, 2>
          }
          %tv1_1, %v1_1 = air.execute -> (memref<8xi8, 2>) {
            %av1_1 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %av1_1 : memref<8xi8, 2>
          }
          %to1_1, %o1_1 = air.execute -> (memref<8xi8, 2>) {
            %ao1_1 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ao1_1 : memref<8xi8, 2>
          }
          %tc1_1 = air.execute [%tw1_1, %tv1_1, %to1_1] {
            linalg.matvec {air.op_cost = "priced"} ins(%w1_1, %v1_1 : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o1_1 : memref<8xi8, 2>)
          }
        }
        %tb1, %b1 = air.execute -> (memref<8xi8, 1>) {
          %ab1 = memref.alloc() : memref<8xi8, 1>
          air.execute_terminator %ab1 : memref<8xi8, 1>
        }
        %p1 = air.channel.put async [%h1, %tb1] @c2[%c0_s, %c0_s] (%b1[] [] []) {id = 201 : i32} : (memref<8xi8, 1>)
        %h2 = air.herd @third async [%tk2]  tile (%x2, %y2) in (%sx2=%c1_s, %sy2=%c4_s) args(%ka2=%k2) : memref<8x8xi8, 1> attributes {id = 12 : i32} {
          %c0_h2 = arith.constant 0 : index
          %tg2, %g2 = air.execute -> (memref<8xi8, 2>) {
            %ag2 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ag2 : memref<8xi8, 2>
          }
          %r2 = air.channel.get async [%tg2] @c2[%c0_h2, %c0_h2] (%g2[] [] []) {id = 102 : i32} : (memref<8xi8, 2>)
          %tw2_0, %w2_0 = air.execute -> (memref<8x8xi8, 2>) {
            %aw2_0 = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %aw2_0 : memref<8x8xi8, 2>
          }
          %tv2_0, %v2_0 = air.execute -> (memref<8xi8, 2>) {
            %av2_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %av2_0 : memref<8xi8, 2>
          }
          %to2_0, %o2_0 = air.execute -> (memref<8xi8, 2>) {
            %ao2_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ao2_0 : memref<8xi8, 2>
          }
          %tc2_0 = air.execute [%tw2_0, %tv2_0, %to2_0] {
            linalg.matvec {air.op_cost = "priced"} ins(%w2_0, %v2_0 : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o2_0 : memref<8xi8, 2>)
          }
        }
        %tb2, %b2 = air.execute -> (memref<8xi8, 1>) {
          %ab2 = memref.alloc() : memref<8xi8, 1>
          air.execute_terminator %ab2 : memref<8xi8, 1>
        }
        %p2 = air.channel.put async [%h2, %tb2] @c3[%c0_s, %c0_s] (%b2[] [] []) {id = 202 : i32} : (memref<8xi8, 1>)
        %h3 = air.herd @fourth async [%tk3]  tile (%x3, %y3) in (%sx3=%c1_s, %sy3=%c2_s) args(%ka3=%k3) : memref<8x8xi8, 1> attributes {id = 13 : i32} {
          %c0_h3 = arith.constant 0 : index
          %tg3, %g3 = air.execute -> (memref<8xi8, 2>) {
            %ag3 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ag3 : memref<8xi8, 2>
          }
          %r3 = air.channel.get async [%tg3] @c3[%c0_h3, %c0_h3] (%g3[] [] []) {id = 103 : i32} : (memref<8xi8, 2>)
          %tw3_0, %w3_0 = air.execute -> (memref<8x8xi8, 2>) {
            %aw3_0 = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %aw3_0 : memref<8x8xi8, 2>
          }
          %tv3_0, %v3_0 = air.execute -> (memref<8xi8, 2>) {
            %av3_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %av3_0 : memref<8xi8, 2>
          }
          %to3_0, %o3_0 = air.execute -> (memref<8xi8, 2>) {
            %ao3_0 = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %ao3_0 : memref<8xi8, 2>
          }
          %tc3_0 = air.execute [%tw3_0, %tv3_0, %to3_0] {
            linalg.matvec {air.op_cost = "priced"} ins(%w3_0, %v3_0 : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o3_0 : memref<8xi8, 2>)
          }
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
