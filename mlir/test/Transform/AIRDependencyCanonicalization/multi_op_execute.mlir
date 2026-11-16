//===- multi_op_execute.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

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
