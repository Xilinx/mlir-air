//===- hoist_herd_same_name_sibling_loops.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests | FileCheck %s

// Herds of one name, each in a loop of its own. Each herd is hoisted out of
// its loop, and the two hoisted herds merge into one that runs both loops.
// Looking for herds of the name beyond the herd's own region would find the
// other loop's herd, which no common region holds, and hoist neither.

// CHECK-LABEL: func.func @same_name_two_loops
// CHECK: air.herd @h
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scf.for
// CHECK: memref.store
// CHECK-NOT: air.herd @h

func.func @same_name_two_loops(%n: i32) {
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
      %r2 = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %r) -> (!air.async.token) {
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
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}
