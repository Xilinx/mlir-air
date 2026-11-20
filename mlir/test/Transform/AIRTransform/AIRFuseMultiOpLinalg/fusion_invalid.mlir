//===- fusion_invalid.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s

// The second op reads the result transposed, so the first op's body cannot be
// inlined as is.

#id = affine_map<(d0, d1) -> (d0, d1)>
#tr = affine_map<(d0, d1) -> (d1, d0)>
func.func @transposed_read(%a: tensor<8x8xf16>) -> tensor<8x8xf32> {
  %e = tensor.empty() : tensor<8x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<8x8xf16>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%in: f16, %o: f32):
    %0 = arith.extf %in : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<8x8xf32>
  %c = linalg.generic {indexing_maps = [#tr, #id], iterator_types = ["parallel", "parallel"]} ins(%p : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %0 = arith.negf %x : f32
    linalg.yield %0 : f32
  } -> tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "transposed_read"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // expected-error @below {{failed to fuse the operations}}
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The first op reads its output, which the fused op has no operand for.

#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @first_op_reads_output(%a: tensor<4x8xf32>, %acc: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  %p = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xf32>) outs(%acc : tensor<4x8xf32>) {
  ^bb0(%in: f32, %o: f32):
    %0 = arith.addf %in, %o : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  %c = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%p : tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %0 = arith.negf %x : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "first_op_reads_output"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match interface{LinalgOp} in %f : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // expected-error @below {{failed to fuse the operations}}
    %fused = transform.air.fuse_multi_op_linalg %first, %second : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
