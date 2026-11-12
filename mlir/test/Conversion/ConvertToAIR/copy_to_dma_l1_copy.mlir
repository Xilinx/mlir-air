//===- copy_to_dma_l1_copy.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter %s | FileCheck %s

// A copy from one L1 buffer to another is the core's own code, not a DMA: it
// becomes a linalg.copy.

// CHECK-LABEL: func.func @l1_copy
// CHECK-NOT: air.dma_memcpy_nd
// CHECK: linalg.copy ins(%arg0 : memref<64xbf16, 2>) outs(%{{.*}} : memref<64xbf16, 2>)
func.func @l1_copy(%arg0: memref<64xbf16, 2>) {
  %b = memref.alloc() : memref<64xbf16, 2>
  memref.copy %arg0, %b : memref<64xbf16, 2> to memref<64xbf16, 2>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %c = transform.structured.match ops{["memref.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %d = transform.air.copy_to_dma %c : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
