//===- fuse_truncf_transform.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.fuse_truncf_linalg

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Fuse truncf into simple elementwise producer
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_into_add"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %producer1 = transform.structured.match ops{["linalg.generic"]} attributes{producer_op} in %func1 : (!pdl.operation) -> !pdl.operation
  %truncf1 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_op} in %func1 : (!pdl.operation) -> !pdl.operation
  %fused1 = transform.air.fuse_truncf_linalg %truncf1, %producer1

  // Test case 2: Fuse truncf into matmul producer
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_into_matmul"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %producer2 = transform.structured.match ops{["linalg.matmul"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %truncf2 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_matmul} in %func2 : (!pdl.operation) -> !pdl.operation
  %fused2 = transform.air.fuse_truncf_linalg %truncf2, %producer2

  // Test case 3: Fuse truncf (f32->bf16) into mul producer
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_f32_to_bf16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %producer3 = transform.structured.match ops{["linalg.generic"]} attributes{producer_mul} in %func3 : (!pdl.operation) -> !pdl.operation
  %truncf3 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_bf16} in %func3 : (!pdl.operation) -> !pdl.operation
  %fused3 = transform.air.fuse_truncf_linalg %truncf3, %producer3

  // Test case 4: Fuse truncf into mixed-precision matmul (bf16 inputs, f32 accumulator)
  // This is the pattern from Triton matmul lowering
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_into_mixed_precision_matmul"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %producer4 = transform.structured.match ops{["linalg.matmul"]} in %func4 : (!pdl.operation) -> !pdl.operation
  %truncf4 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_mixed_matmul} in %func4 : (!pdl.operation) -> !pdl.operation
  %fused4 = transform.air.fuse_truncf_linalg %truncf4, %producer4

  // Test case 5: Fuse truncf with bufferization pattern
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_truncf_bufferized"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %producer5 = transform.structured.match ops{["linalg.matmul"]} in %func5 : (!pdl.operation) -> !pdl.operation
  %truncf5 = transform.structured.match ops{["linalg.generic"]} attributes{truncf_bufferized} in %func5 : (!pdl.operation) -> !pdl.operation
  %fused5 = transform.air.fuse_truncf_linalg %truncf5, %producer5
}
