//===- air_translate_invalid.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --split-input-file --verify-diagnostics %s

// This rank's own symmetric buffer is rank %rx's; translating it from the
// constant rank 1 is right on rank 1 only.
func.func @translate_constant_from(%bases_outer : memref<?xindex>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%bases = %bases_outer) : memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf = memref.alloc() {air.symmetric} : memref<128xf32>
    // expected-error @+1 {{'air.translate' op source is this rank's own air.symmetric buffer, but from_rank is the constant 1, which is this rank only on one of the 2 ranks}}
    %peer = air.translate %buf, %c1, %c0, %bases : memref<128xf32>, memref<?xindex>
  }
  return
}

// -----

// Translated from the rank's own id: accepted.
func.func @translate_from_rank_id(%bases_outer : memref<?xindex>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%bases = %bases_outer) : memref<?xindex> {
    %c0 = arith.constant 0 : index
    %buf = memref.alloc() {air.symmetric} : memref<128xf32>
    %peer = air.translate %buf, %rx, %c0, %bases : memref<128xf32>, memref<?xindex>
  }
  return
}
