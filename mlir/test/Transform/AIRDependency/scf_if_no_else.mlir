//===- scf_if_no_else.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Test that air-dependency handles scf.if without else block (no crash on
// empty region) and that memref.copy inside scf.if is not wrapped in
// air.execute (which would create broken async chains).

// CHECK-LABEL: func.func @scf_if_no_else
// CHECK: air.herd
// CHECK: memref.alloc
// CHECK: memref.alloc
// CHECK: scf.if
// CHECK:   memref.copy
// CHECK: air.execute
// CHECK:   memref.dealloc
module {
  func.func @scf_if_no_else() {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %true = arith.constant true
      %alloc_a = memref.alloc() : memref<48xi32, 2 : i32>
      %alloc_b = memref.alloc() : memref<48xi32, 2 : i32>
      // scf.if WITHOUT else block — must not crash (empty region guard)
      scf.if %true {
        // memref.copy inside scf.if — must not be wrapped in air.execute
        memref.copy %alloc_a, %alloc_b : memref<48xi32, 2 : i32> to memref<48xi32, 2 : i32>
      }
      memref.dealloc %alloc_a : memref<48xi32, 2 : i32>
      memref.dealloc %alloc_b : memref<48xi32, 2 : i32>
    }
    return
  }
}
