//===- fuse_truncf_transform.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.fuse_truncf_linalg

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Fuse truncf into simple elementwise producer
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_into_add"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %producer1 = transform.structured.match ops{["linalg.generic"]} attributes{producer_op} in %func1 : (!transform.any_op) -> !transform.any_op
  %truncf1 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_op} in %func1 : (!transform.any_op) -> !transform.any_op
  %fused1 = transform.air.fuse_truncf_linalg %truncf1, %producer1 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 2: Fuse truncf into matmul producer
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_into_matmul"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %producer2 = transform.structured.match ops{["linalg.matmul"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %truncf2 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_matmul} in %func2 : (!transform.any_op) -> !transform.any_op
  %fused2 = transform.air.fuse_truncf_linalg %truncf2, %producer2 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 3: Fuse truncf (f32->bf16) into mul producer
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_f32_to_bf16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %producer3 = transform.structured.match ops{["linalg.generic"]} attributes{producer_mul} in %func3 : (!transform.any_op) -> !transform.any_op
  %truncf3 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_bf16} in %func3 : (!transform.any_op) -> !transform.any_op
  %fused3 = transform.air.fuse_truncf_linalg %truncf3, %producer3 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 4: Fuse truncf into mixed-precision matmul (bf16 inputs, f32 accumulator)
  // This is the pattern from Triton matmul lowering
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_into_mixed_precision_matmul"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %producer4 = transform.structured.match ops{["linalg.matmul"]} in %func4 : (!transform.any_op) -> !transform.any_op
  %truncf4 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_mixed_matmul} in %func4 : (!transform.any_op) -> !transform.any_op
  %fused4 = transform.air.fuse_truncf_linalg %truncf4, %producer4 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 5: Fuse truncf with bufferization pattern
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_bufferized"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %producer5 = transform.structured.match ops{["linalg.matmul"]} in %func5 : (!transform.any_op) -> !transform.any_op
  %truncf5 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_bufferized} in %func5 : (!transform.any_op) -> !transform.any_op
  %fused5 = transform.air.fuse_truncf_linalg %truncf5, %producer5 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
