//===- hoist_herd_live_ins.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests -split-input-file | FileCheck %s

// A herd operand computed in the loop the herd is hoisted out of, from the
// loop's induction variable and from values defined outside the loop. Those
// values become herd operands, and the computation stays ahead of the herd
// body's ops, which run in the loop inside the herd.
// CHECK-LABEL: func.func @operand_from_loop
// CHECK: air.segment @seg async args(%[[SN:.*]]=%{{.*}}, %[[SA:.*]]=%{{.*}}) : i32, i32
// CHECK: air.herd @herd_0 async {{.*}} args(%[[HA:.*]]=%[[SA]], %[[HN:.*]]=%[[SN]]) : i32, i32
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK: %[[ADD:.*]] = arith.addi %[[HA]], %[[IV]] : i32
// CHECK: %[[SUB:.*]] = arith.subi %[[HN]], %[[ADD]] : i32
// CHECK: memref.store %[[SUB]]
func.func @operand_from_loop(%n: i32, %a: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n, %la=%a) : i32, i32 {
    %s = air.segment @seg async args(%sn=%ln, %sa=%la) : i32, i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %iv = arith.index_cast %i : index to i32
        %add = arith.addi %sa, %iv : i32
        %sub = arith.subi %sn, %add : i32
        %h = air.herd @herd_0 async [%d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sub) : i32 {
          %c0_1 = arith.constant 0 : index
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %al = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %al : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.store %q, %b[%c0_1] : memref<16xi32, 2>
          }
          %t2 = air.execute [%t1] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}

// -----

// As above, with the operand computed in the inner loop from a value the
// outer loop computes. The herd is hoisted out of both loops.
// CHECK-LABEL: func.func @operand_from_outer_loop
// CHECK: air.segment @seg async args(%[[SN:.*]]=%{{.*}}) : i32
// CHECK: air.herd @herd_0 async {{.*}} args(%[[HN:.*]]=%[[SN]]) : i32
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK: %[[BASE:.*]] = arith.muli %[[IV]], %[[HN]] : i32
// CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: %[[JV:.*]] = arith.index_cast %[[J]] : index to i32
// CHECK: %[[ADD:.*]] = arith.addi %[[BASE]], %[[JV]] : i32
// CHECK: memref.store %[[ADD]]
func.func @operand_from_outer_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %c2 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %iv = arith.index_cast %i : index to i32
        %base = arith.muli %iv, %sn : i32
        %r2 = scf.for %j = %c0 to %c4 step %c1_0 iter_args(%d2 = %d) -> (!air.async.token) {
          %jv = arith.index_cast %j : index to i32
          %q0 = arith.addi %base, %jv : i32
          %h = air.herd @herd_0 async [%d2] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%q0) : i32 {
            %c0_1 = arith.constant 0 : index
            %t0, %b = air.execute -> (memref<16xi32, 2>) {
              %al = memref.alloc() : memref<16xi32, 2>
              air.execute_terminator %al : memref<16xi32, 2>
            }
            %t1 = air.execute [%t0] {
              memref.store %q, %b[%c0_1] : memref<16xi32, 2>
            }
            %t2 = air.execute [%t1] {
              memref.dealloc %b : memref<16xi32, 2>
            }
          }
          scf.yield %h : !air.async.token
        }
        scf.yield %r2 : !air.async.token
      }
    }
  }
  return
}

// -----

// The operand is computed from a constant defined outside the loop. The
// constant is copied into the herd rather than passed in: the herd takes no
// operands.
// CHECK-LABEL: func.func @constant_operand
// CHECK: air.herd @herd_0 async tile ({{.*}}) in ({{.*}}) {
// CHECK: %[[C3:.*]] = arith.constant 3 : i32
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: %[[IV:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK: %[[MUL:.*]] = arith.muli %[[IV]], %[[C3]] : i32
// CHECK: memref.store %[[MUL]]
func.func @constant_operand(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c3_i32 = arith.constant 3 : i32
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %iv = arith.index_cast %i : index to i32
        %mul = arith.muli %iv, %c3_i32 : i32
        %h = air.herd @herd_0 async [%d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%mul) : i32 {
          %c0_1 = arith.constant 0 : index
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %al = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %al : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.store %q, %b[%c0_1] : memref<16xi32, 2>
          }
          %t2 = air.execute [%t1] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}

// -----

// The loop the herd is hoisted out of computes the herd's operand in an
// air.execute that waits for an L2 allocation made before the loop. The herd
// takes that wait over: it starts once the allocation is done, and inside it
// the execute no longer waits for anything outside.

// CHECK-LABEL: func.func @outside_token
// CHECK: %[[TOK:.*]], %{{.*}} = air.execute -> (memref<16xi32, 1>)
// CHECK: air.herd @herd_0 async [%[[TOK]]]
// CHECK: scf.for
// CHECK: air.execute -> (i32)
// CHECK: arith.addi
func.func @outside_token(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %e, %l2 = air.execute -> (memref<16xi32, 1>) {
        %a2 = memref.alloc() : memref<16xi32, 1>
        air.execute_terminator %a2 : memref<16xi32, 1>
      }
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %x, %v = air.execute [%e] -> (i32) {
          %iv = arith.index_cast %i : index to i32
          %add = arith.addi %sn, %iv : i32
          air.execute_terminator %add : i32
        }
        %h = air.herd @herd_0 async [%d, %x] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%v) : i32 {
          %c0_1 = arith.constant 0 : index
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %al = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %al : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.store %q, %b[%c0_1] : memref<16xi32, 2>
          }
          %t2 = air.execute [%t1] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}

// -----

// The herd itself waits for a channel put made before the loop. Only the
// herd uses that token, which goes away with it, so the herd is hoisted.

// CHECK-LABEL: func.func @herd_waits_on_channel
// CHECK: air.channel.put
// CHECK-NEXT: air.herd @herd_0
// CHECK: scf.for

air.channel @feed [1, 1]
func.func @herd_waits_on_channel(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %ta, %l2 = air.execute -> (memref<16xi32, 1>) {
        %a2 = memref.alloc() : memref<16xi32, 1>
        air.execute_terminator %a2 : memref<16xi32, 1>
      }
      %e = air.channel.put async [%ta] @feed[] (%l2[] [] []) : (memref<16xi32, 1>)
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %h = air.herd @herd_0 async [%d, %e] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sn) : i32 {
          %c0_1 = arith.constant 0 : index
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %al = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %al : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.store %q, %b[%c0_1] : memref<16xi32, 2>
          }
          %t2 = air.execute [%t1] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}
