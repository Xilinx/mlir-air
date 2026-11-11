//===- unshare_empties.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter %s | FileCheck %s

// Two fills write into one tensor.empty: each takes an empty of its own, so
// they bufferize to two buffers rather than one the second overwrites.

// CHECK-LABEL: func.func @two_fills
// CHECK-DAG: %[[E0:.*]] = tensor.empty() : tensor<64xf32>
// CHECK-DAG: %[[E1:.*]] = tensor.empty() : tensor<64xf32>
// CHECK-DAG: linalg.fill ins(%arg0 : f32) outs(%[[E1]] : tensor<64xf32>)
// CHECK-DAG: linalg.fill ins(%arg1 : f32) outs(%[[E0]] : tensor<64xf32>)
func.func @two_fills(%a: f32, %b: f32) -> (tensor<64xf32>, tensor<64xf32>) {
  %e = tensor.empty() : tensor<64xf32>
  %x = linalg.fill ins(%a : f32) outs(%e : tensor<64xf32>) -> tensor<64xf32>
  %y = linalg.fill ins(%b : f32) outs(%e : tensor<64xf32>) -> tensor<64xf32>
  return %x, %y : tensor<64xf32>, tensor<64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %g = transform.air.unshare_empties %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
