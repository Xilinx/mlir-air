//===- loop_in_execute_vertex.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// A loop inside an air.execute has a vertex of its own and is also the first
// op of the execute's chain. Edges into the execute land on the chain, so the
// execute keeps its dependency, and the loop's own vertex is still found by
// its id, whatever ids the ops before it took.

// CHECK-LABEL: func.func @execute_then_loop_in_execute
// CHECK: %[[T0:.*]], %{{.*}} = air.execute [%{{.*}}] -> (memref<4xi32>)
// CHECK: %[[T1:.*]] = air.execute [%[[T0]]] {
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[T0]])
// CHECK: air.wait_all async [%[[T1]]]
func.func @execute_then_loop_in_execute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %t0 = air.wait_all async
  %t1, %r = air.execute [%t0] -> (memref<4xi32>) {
    %m = memref.alloc() : memref<4xi32>
    air.execute_terminator %m : memref<4xi32>
  }
  %t2 = air.execute [%t1] {
    %f = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %t1) -> (!air.async.token) {
      %w = air.wait_all async [%a]
      scf.yield %w : !air.async.token
    }
    air.execute_terminator
  }
  %t3 = air.wait_all async [%t1, %t2]
  return
}

// The same function after one with executes of its own: ids no longer line
// up by chance.

// CHECK-LABEL: func.func @two_ops
// CHECK-LABEL: func.func @loop_in_execute
// CHECK: %[[W:.*]] = air.wait_all async
// CHECK: air.execute [%[[W]]] {
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])
func.func @two_ops() {
  %t0 = air.wait_all async
  %t1, %r1 = air.execute [%t0] -> (memref<4xi32>) {
    %m = memref.alloc() : memref<4xi32>
    air.execute_terminator %m : memref<4xi32>
  }
  %t2 = air.execute [%t1] {
    memref.dealloc %r1 : memref<4xi32>
    air.execute_terminator
  }
  %t3 = air.wait_all async [%t1, %t2]
  return
}

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
