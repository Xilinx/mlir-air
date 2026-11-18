//===- air_loop_merging_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-loop-merging="loop-merge-levels=0,2 air-label=band" -split-input-file -verify-diagnostics

// Only consecutive loop levels can be merged.
func.func @gap(%m: memref<2x3x4xf32>) {
  // expected-error @+1 {{loop-merge-levels must be consecutive}}
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      affine.for %k = 0 to 4 {
        %v = affine.load %m[%i, %j, %k] : memref<2x3x4xf32>
        affine.store %v, %m[%i, %j, %k] : memref<2x3x4xf32>
      }
    }
  } {affine_opt_label = "band"}
  return
}
