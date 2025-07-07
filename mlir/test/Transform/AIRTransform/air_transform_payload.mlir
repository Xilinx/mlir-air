//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform_ops.mlir' %s | FileCheck %s

// CHECK: scf.forall
func.func @mmult(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%C : memref<1024x1024xf32>)
  linalg.matmul ins(%A, %B : memref<1024x1024xf32>, memref<1024x1024xf32>) outs(%C : memref<1024x1024xf32>)
  return %C : memref<1024x1024xf32>
}
