//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.fuse_extf_linalg

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic fusion of extf with elementwise add
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_extf_with_add"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops1 = transform.structured.match ops{["linalg.generic"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %extf_op1, %consumer_op1 = transform.split_handle %ops1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused1 = transform.air.fuse_extf_linalg %extf_op1, %consumer_op1

  // Test case 2: Fusion with multiple inputs where extf result is not the first input
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_extf_with_mul_second_input"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops2 = transform.structured.match ops{["linalg.generic"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %extf_op2, %consumer_op2 = transform.split_handle %ops2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused2 = transform.air.fuse_extf_linalg %extf_op2, %consumer_op2

  // Test case 3: 2D tensor fusion
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_extf_2d_tensor"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops3 = transform.structured.match ops{["linalg.generic"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %extf_op3, %consumer_op3 = transform.split_handle %ops3 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused3 = transform.air.fuse_extf_linalg %extf_op3, %consumer_op3

  // Test case 4: Different precision extension (bf16 to f32)
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_extf_bf16_to_f32"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops4 = transform.structured.match ops{["linalg.generic"]} in %func4 : (!pdl.operation) -> !pdl.operation
  %extf_op4, %consumer_op4 = transform.split_handle %ops4 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused4 = transform.air.fuse_extf_linalg %extf_op4, %consumer_op4
}
