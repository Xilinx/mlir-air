//===- memref_store.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Test that memref.store inside a herd body is correctly tracked by the
// dependency pass. The store should be wrapped in air.execute and not
// eliminated as dead code. This validates the fix in Dependency.cpp that
// correctly classifies memref.store as writing only to the memref operand.

// CHECK-LABEL: func.func @memref_store_in_herd
// CHECK: air.herd
// CHECK: air.execute {
// CHECK-NEXT: memref.store
// CHECK-NEXT: }
func.func @memref_store_in_herd(%buf : memref<32xi8, 2>) {
  %c1 = arith.constant 1 : index
  air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%arg0=%buf) : memref<32xi8, 2> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 42 : i8
    memref.store %cst, %arg0[%c0] : memref<32xi8, 2>
    air.herd_terminator
  }
  return
}
