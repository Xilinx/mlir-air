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

// -----

// Only rank 1 runs the translation, so translating from rank 1 is right: the
// constant is checked only directly in the rank's body.
func.func @translate_guarded_by_rank(%bases_outer : memref<?xindex>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%bases = %bases_outer) : memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf = memref.alloc() {air.symmetric} : memref<128xf32>
    %is1 = arith.cmpi eq, %rx, %c1 : index
    scf.if %is1 {
      %peer = air.translate %buf, %c1, %c0, %bases : memref<128xf32>, memref<?xindex>
    }
  }
  return
}

// -----

// A number of ranks whose product does not fit is not bounded.
func.func @rank_count_overflow() {
  %big = arith.constant 4294967296 : index
  air.rank (%rx, %ry) in (%sx = %big, %sy = %big) {
    %dst = memref.alloc() : memref<128xf32, 2>
    %src = memref.alloc() {air.symmetric} : memref<128xf32>
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {src_rank = 5 : i64}
        : (memref<128xf32, 2>, memref<128xf32>)
  }
  return
}
