//===- air_translate.mlir - air.translate parser, printer, folder --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s
// RUN: air-opt --canonicalize %s | FileCheck %s --check-prefix=FOLD

// Round-trip: 1D static memref.
// CHECK-LABEL: func.func @translate_1d
// CHECK: %{{.*}} = air.translate %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<1024xf32>, memref<?xindex>
func.func @translate_1d(%src : memref<1024xf32>, %from : index, %to : index, %bases : memref<?xindex>) -> memref<1024xf32> {
  %peer = air.translate %src, %from, %to, %bases : memref<1024xf32>, memref<?xindex>
  return %peer : memref<1024xf32>
}

// Round-trip: 2D static memref in address space 1.
// CHECK-LABEL: func.func @translate_2d_addrspace
// CHECK: air.translate %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<64x64xf32, 1>, memref<?xindex>
func.func @translate_2d_addrspace(%src : memref<64x64xf32, 1>, %from : index, %to : index, %bases : memref<?xindex>) -> memref<64x64xf32, 1> {
  %peer = air.translate %src, %from, %to, %bases : memref<64x64xf32, 1>, memref<?xindex>
  return %peer : memref<64x64xf32, 1>
}

// Round-trip: static-shaped heap_bases is also accepted.
// CHECK-LABEL: func.func @translate_static_bases
// CHECK: air.translate %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<8xf32>, memref<8xindex>
func.func @translate_static_bases(%src : memref<8xf32>, %from : index, %to : index, %bases : memref<8xindex>) -> memref<8xf32> {
  %peer = air.translate %src, %from, %to, %bases : memref<8xf32>, memref<8xindex>
  return %peer : memref<8xf32>
}

// Folder: from_rank == to_rank (same SSA value) folds to %src.
// FOLD-LABEL: func.func @fold_same_rank
// FOLD-NOT: air.translate
// FOLD: return %arg0 : memref<8xf32>
func.func @fold_same_rank(%src : memref<8xf32>, %r : index, %bases : memref<?xindex>) -> memref<8xf32> {
  %peer = air.translate %src, %r, %r, %bases : memref<8xf32>, memref<?xindex>
  return %peer : memref<8xf32>
}

// Folder: distinct constants with same value also fold.
// FOLD-LABEL: func.func @fold_constant_eq_ranks
// FOLD-NOT: air.translate
// FOLD: return %arg0 : memref<8xf32>
func.func @fold_constant_eq_ranks(%src : memref<8xf32>, %bases : memref<?xindex>) -> memref<8xf32> {
  %c2 = arith.constant 2 : index
  %c2_again = arith.constant 2 : index
  %peer = air.translate %src, %c2, %c2_again, %bases : memref<8xf32>, memref<?xindex>
  return %peer : memref<8xf32>
}

// Non-fold: distinct constants do NOT fold.
// FOLD-LABEL: func.func @no_fold_distinct_constants
// FOLD: air.translate
func.func @no_fold_distinct_constants(%src : memref<8xf32>, %bases : memref<?xindex>) -> memref<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %peer = air.translate %src, %c0, %c1, %bases : memref<8xf32>, memref<?xindex>
  return %peer : memref<8xf32>
}
