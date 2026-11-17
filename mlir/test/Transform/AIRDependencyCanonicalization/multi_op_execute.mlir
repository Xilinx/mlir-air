//===- multi_op_execute.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-dependency-canonicalize | FileCheck %s

// Edges leave an air.execute from the vertex of its terminator. With two or
// more ops in the body, that vertex holds the terminator rather than the
// execute, and the dependencies on the execute were dropped.

// CHECK-LABEL: func.func @two_ops
// CHECK: %[[T1:.*]] = air.execute [%{{.*}}] {
// CHECK: air.wait_all async [%[[T1]]]
func.func @two_ops() {
  %t0 = air.execute {
    air.execute_terminator
  }
  %t1 = air.execute [%t0] {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    air.execute_terminator
  }
  %t2 = air.wait_all async [%t1]
  return
}

// -----

// %t2 reads %v0, so it must stay ordered after %t0, here through %t1.
// CHECK-LABEL: func.func @reads_result
// CHECK: %[[T0:.*]], %{{.*}} = air.execute -> (i32)
// CHECK: %[[T1:.*]], %{{.*}} = air.execute [%[[T0]]] -> (i32)
// CHECK: air.execute [%[[T1]]] -> (i32)
func.func @reads_result() {
  %t0, %v0 = air.execute -> (i32) {
    %c1 = arith.constant 1 : i32
    air.execute_terminator %c1 : i32
  }
  %t1, %v1 = air.execute [%t0] -> (i32) {
    %c10 = arith.constant 10 : i32
    %s = arith.addi %c10, %v0 : i32
    air.execute_terminator %s : i32
  }
  %t2, %v2 = air.execute [%t0, %t1] -> (i32) {
    %c100 = arith.constant 100 : i32
    %s2 = arith.addi %c100, %v0 : i32
    air.execute_terminator %s2 : i32
  }
  return
}

// -----

// A loop in an execute's body has a vertex of its own: an op in the loop
// keeps its dependency on the loop-carried token.
// CHECK-LABEL: func.func @loop_in_execute
// CHECK: scf.for %{{.*}} iter_args(%[[A:.*]] = %{{.*}})
// CHECK-NEXT: air.wait_all async [%[[A]]]
func.func @loop_in_execute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %t0 = air.wait_all async
  %t1 = air.execute [%t0] {
    %f = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %t0) -> (!air.async.token) {
      %w = air.wait_all async [%a]
      scf.yield %w : !air.async.token
    }
    air.execute_terminator
  }
  return
}
