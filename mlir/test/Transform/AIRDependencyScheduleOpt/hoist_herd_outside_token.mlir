//===- hoist_herd_outside_token.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests | FileCheck %s

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
