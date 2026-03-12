//===- linalg_promote_multi_op.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that linalg_promote correctly promotes multiple linalg ops that share
// the same input subview (e.g., weighted_rms_norm pattern where sq and out
// generics both read from the same X slice) and a broadcast operand.
//
// Verifies:
// 1. No SSA domination errors
// 2. Broadcast operand promoted to target memory space
// 3. No redundant L1→L1 copies from memory space attribute mismatch

// RUN: air-opt %s -air-transform='filename=%s' | FileCheck %s

// CHECK-LABEL: func.func @multi_op_shared_input
// CHECK: memref.alloc() : memref<4x64xf32, 2 : i32>
// CHECK: memref.copy
// CHECK: memref.alloc() : memref<64xf32, 2 : i32>
// CHECK: memref.copy
// CHECK-NOT: memref<?x?xf32
func.func @multi_op_shared_input(
    %input: memref<16x64xf32, 1>,
    %weight: memref<64xf32, 1>,
    %output: memref<16x64xf32, 1>) {
  %c0 = arith.constant 0 : index
  %subview_in = memref.subview %input[%c0, 0] [4, 64] [1, 1]
      : memref<16x64xf32, 1> to memref<4x64xf32, strided<[64, 1], offset: ?>, 1>
  %alloc_sq = memref.alloc() : memref<4x64xf32>
  // Op 1: square (reads from subview_in)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%subview_in : memref<4x64xf32, strided<[64, 1], offset: ?>, 1>)
    outs(%alloc_sq : memref<4x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sq = arith.mulf %in, %in : f32
    linalg.yield %sq : f32
  }
  // Op 2: reduce (reads from alloc_sq)
  %alloc_sum = memref.alloc() : memref<4xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc_sum : memref<4xf32>)
  linalg.reduce ins(%alloc_sq : memref<4x64xf32>)
                outs(%alloc_sum : memref<4xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %add = arith.addf %in, %init : f32
      linalg.yield %add : f32
    }
  %subview_out = memref.subview %output[%c0, 0] [4, 64] [1, 1]
      : memref<16x64xf32, 1> to memref<4x64xf32, strided<[64, 1], offset: ?>, 1>
  // Op 3: normalize with broadcast weight (reads from subview_in and weight)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%subview_in, %alloc_sum, %weight
      : memref<4x64xf32, strided<[64, 1], offset: ?>, 1>,
        memref<4xf32>, memref<64xf32, 1>)
    outs(%subview_out
      : memref<4x64xf32, strided<[64, 1], offset: ?>, 1>) {
  ^bb0(%x: f32, %sum: f32, %w: f32, %out: f32):
    %norm = arith.divf %x, %sum : f32
    %weighted = arith.mulf %norm, %w : f32
    linalg.yield %weighted : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic", "linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.air.linalg_promote %0 {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
