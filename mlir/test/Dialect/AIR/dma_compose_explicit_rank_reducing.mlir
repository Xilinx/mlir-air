//===- dma_compose_explicit_rank_reducing.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize="test-convergence" %s | FileCheck %s

// An explicit partial access through a rank-reducing subview that drops a
// dim at a nonzero offset is not composed, and canonicalization converges
// without touching it.

// CHECK-LABEL: func.func @explicit_through_rank_reducing
// CHECK: %[[SV:.*]] = memref.subview
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[SV]][%arg0] [4] [1])
func.func @explicit_through_rank_reducing(%i: index) {
  %0 = memref.alloc() : memref<4x8xi16, 1>
  %1 = memref.alloc() : memref<4xi16, 2>
  %sv = memref.subview %0[1, 0] [1, 8] [1, 1] : memref<4x8xi16, 1> to memref<8xi16, strided<[1], offset: 8>, 1>
  air.dma_memcpy_nd (%1[] [] [], %sv[%i] [4] [1]) : (memref<4xi16, 2>, memref<8xi16, strided<[1], offset: 8>, 1>)
  return
}
