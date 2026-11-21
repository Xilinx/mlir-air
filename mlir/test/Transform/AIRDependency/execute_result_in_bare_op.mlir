//===- execute_result_in_bare_op.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// A loop bound loaded from memory: the load becomes an air.execute, and the
// index_cast and the loop read its result when the host reaches them. A
// blocking wait on the load's token comes before them.

// CHECK-LABEL: func.func @device_bound
// CHECK: %[[T:.*]], %[[V:.*]] = air.execute -> (i32) {
// CHECK-NEXT: memref.load
// CHECK: air.wait_all [%[[T]]]
// CHECK-NEXT: %[[K:.*]] = arith.index_cast %[[V]]
// CHECK: scf.for {{.*}} to %[[K]]
func.func @device_bound(%n: memref<1xi32>, %buf: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : i32
  %v = memref.load %n[%c0] : memref<1xi32>
  %k = arith.index_cast %v : i32 to index
  scf.for %i = %c0 to %k step %c1 {
    memref.store %c7, %buf[%i] : memref<8xi32>
  }
  return
}
