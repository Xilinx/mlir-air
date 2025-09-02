//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @air_fuse_into
// CHECK: scf.forall
// CHECK: linalg.fill
// CHECK: linalg.matmul
#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @air_fuse_into(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %D: memref<128x128xf32>) -> memref<128x128xf32>
{
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%D : memref<128x128xf32>)
  linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>) outs(%D : memref<128x128xf32>)
  return %D : memref<128x128xf32>
}
