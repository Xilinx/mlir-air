//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// CHECK: memref.alloc()
// CHECK: scf.for
// CHECK: }
// CHECK: memref.dealloc
func.func @func0(%arg0: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0 : i32
  scf.for %i = %c0 to %c4 step %c1 {
    %tmp = memref.alloc() : memref<64xi32>
    linalg.fill ins(%cst : i32) outs(%tmp : memref<64xi32>)
    memref.dealloc %tmp : memref<64xi32>
  }
  return
}
