//===- air_transform_loop_payload.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_loop.mlir' %s | FileCheck %s

// The target need not be isolated from above: only the tagged loop's body is
// linearized, and the elementwise op outside it keeps its n-D type.

// CHECK-LABEL: @scoped
// CHECK: scf.for
// CHECK: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
// CHECK-COUNT-2: arith.mulf {{.*}} : vector<16xf32>
// CHECK: {linearize}
// CHECK: arith.addf {{.*}} : vector<4x8xf32>
func.func @scoped(%a: memref<4x4x8xf32>, %s: memref<4x1x8xf32>, %b: memref<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %p = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %av = vector.transfer_read %a[%i, %c0, %c0], %p {in_bounds = [true, true, true]} : memref<4x4x8xf32>, vector<1x4x8xf32>
    %sv = vector.transfer_read %s[%i, %c0, %c0], %p {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, 0, d2)>} : memref<4x1x8xf32>, vector<1x4x8xf32>
    %r = arith.mulf %av, %sv : vector<1x4x8xf32>
    vector.transfer_write %r, %a[%i, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x8xf32>, memref<4x4x8xf32>
  } {linearize}
  %bv = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
  %d = arith.addf %bv, %bv : vector<4x8xf32>
  vector.transfer_write %d, %b[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
  return
}
