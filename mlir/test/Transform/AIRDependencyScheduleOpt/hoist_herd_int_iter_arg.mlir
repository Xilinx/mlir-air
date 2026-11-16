//===- hoist_herd_int_iter_arg.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests | FileCheck %s

// Herds of one name in sibling loops, the second loop also carrying an i32.
// Only token iter_args can be carried through a hoisted nest: the first herd
// is hoisted, the second is left in its loop, unchanged.

// CHECK-LABEL: func.func @int_iter_arg
// CHECK: air.herd @h
// CHECK: scf.for
// CHECK: scf.for {{.*}} -> (!air.async.token, i32)
// CHECK-NEXT: air.herd @h

func.func @int_iter_arg(%n: i32) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : i32 {
    %s = air.segment @seg async args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %h = air.herd @h async [%d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sn) : i32 {
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
      %c0_i32 = arith.constant 0 : i32
      %r2:2 = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %r, %acc = %c0_i32) -> (!air.async.token, i32) {
        %h = air.herd @h async [%d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%q=%sn) : i32 {
          %c1_1 = arith.constant 1 : index
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %al = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %al : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.store %q, %b[%c1_1] : memref<16xi32, 2>
          }
          %t2 = air.execute [%t1] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        %acc1 = arith.addi %acc, %sn : i32
        scf.yield %h, %acc1 : !air.async.token, i32
      }
    }
  }
  return
}
