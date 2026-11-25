//===- translate_after_rank_to_launch.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-rank-to-launch | FileCheck %s

// The lowering removes the air.rank and keeps the symmetric allocation and the
// air.translate; the result still verifies.

// CHECK-LABEL: func.func @translate_through_lowering
// CHECK-NOT: air.rank
// CHECK: memref.alloc() {air.symmetric}
// CHECK: air.translate
func.func @translate_through_lowering(%bases : memref<?xindex>) {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) args(%b = %bases) : memref<?xindex> {
    %c0 = arith.constant 0 : index
    %buf = memref.alloc() {air.symmetric} : memref<128xf32>
    %peer = air.translate %buf, %rx, %c0, %b : memref<128xf32>, memref<?xindex>
    memref.dealloc %buf : memref<128xf32>
  }
  return
}
