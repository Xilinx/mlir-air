//===- air_loop_merging_dependent_bound.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-loop-merging="loop-merge-levels=0,1 air-label=band" -verify-diagnostics

// The merged loops must have constant bounds; here the inner bound depends on
// the outer induction variable.
func.func @dependent_bound(%m: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    // expected-error @+1 {{non-static loop bounds}}
    affine.for %j = 0 to affine_map<(d0) -> (d0)>(%i) {
      %v = affine.load %m[%i, %j] : memref<4x4xf32>
      affine.store %v, %m[%i, %j] : memref<4x4xf32>
    }
  } {affine_opt_label = "band"}
  return
}
