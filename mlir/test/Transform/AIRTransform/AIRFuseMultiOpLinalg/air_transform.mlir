//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.fuse_multi_op_linalg

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Fuse extf + mulf with reduction
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_extf_mulf_with_reduce"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops1 = transform.structured.match ops{["linalg.generic"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %first_op1, %consumer_op1 = transform.split_handle %ops1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused1 = transform.air.fuse_multi_op_linalg %first_op1, %consumer_op1

  // Test case 2: Fuse single extf (backward compatibility)
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_single_extf"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops2 = transform.structured.match ops{["linalg.generic"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %first_op2, %consumer_op2 = transform.split_handle %ops2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused2 = transform.air.fuse_multi_op_linalg %first_op2, %consumer_op2

  // Test case 3: Fuse with math operations (extf + sqrt + mulf)
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_with_math_ops"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops3 = transform.structured.match ops{["linalg.generic"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %first_op3, %consumer_op3 = transform.split_handle %ops3 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused3 = transform.air.fuse_multi_op_linalg %first_op3, %consumer_op3

  // Test case 4: 2D tensors with multiple operations
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_2d_multi_ops"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops4 = transform.structured.match ops{["linalg.generic"]} in %func4 : (!pdl.operation) -> !pdl.operation
  %first_op4, %consumer_op4 = transform.split_handle %ops4 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused4 = transform.air.fuse_multi_op_linalg %first_op4, %consumer_op4

  // Test case 5: Multi-input first op with reduction (real-world softmax pattern)
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_multi_input_with_reduce"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %ops5 = transform.structured.match ops{["linalg.generic"]} in %func5 : (!pdl.operation) -> !pdl.operation
  %first_op5, %consumer_op5 = transform.split_handle %ops5 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %fused5 = transform.air.fuse_multi_op_linalg %first_op5, %consumer_op5
}
