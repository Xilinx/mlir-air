//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.fuse_multi_op_linalg

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Fuse extf + mulf with reduction
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_extf_mulf_with_reduce"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %ops1 = transform.structured.match ops{["linalg.generic"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %first_op1, %consumer_op1 = transform.split_handle %ops1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused1 = transform.air.fuse_multi_op_linalg %first_op1, %consumer_op1 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 2: Fuse single extf (backward compatibility)
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_single_extf"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %ops2 = transform.structured.match ops{["linalg.generic"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %first_op2, %consumer_op2 = transform.split_handle %ops2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused2 = transform.air.fuse_multi_op_linalg %first_op2, %consumer_op2 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 3: Fuse with math operations (extf + sqrt + mulf)
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_with_math_ops"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %ops3 = transform.structured.match ops{["linalg.generic"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %first_op3, %consumer_op3 = transform.split_handle %ops3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused3 = transform.air.fuse_multi_op_linalg %first_op3, %consumer_op3 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 4: 2D tensors with multiple operations
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_2d_multi_ops"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %ops4 = transform.structured.match ops{["linalg.generic"]} in %func4 : (!transform.any_op) -> !transform.any_op
  %first_op4, %consumer_op4 = transform.split_handle %ops4 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused4 = transform.air.fuse_multi_op_linalg %first_op4, %consumer_op4 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 5: Multi-input first op with reduction (real-world softmax pattern)
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_multi_input_with_reduce"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %ops5 = transform.structured.match ops{["linalg.generic"]} in %func5 : (!transform.any_op) -> !transform.any_op
  %first_op5, %consumer_op5 = transform.split_handle %ops5 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %fused5 = transform.air.fuse_multi_op_linalg %first_op5, %consumer_op5 : (!transform.any_op, !transform.any_op) -> !transform.any_op

  // Test case 6: linalg.generic + linalg.reduce (per-handle generalize pattern)
  // Tests the AIE2P softmax pattern: data-flow navigation captures linalg.reduce
  // as typed anchor, then generalize is applied per-handle before fusion.
  %func6 = transform.structured.match ops{["func.func"]} attributes{sym_name = "fuse_generic_with_reduce"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %generic6 = transform.structured.match ops{["linalg.generic"]} in %func6 : (!transform.any_op) -> !transform.any_op
  %reduce6 = transform.structured.match ops{["linalg.reduce"]} in %func6 : (!transform.any_op) -> !transform.any_op
  // Per-handle generalize: convert linalg.reduce to linalg.generic just before fusion
  %reduce6_gen = transform.structured.generalize %reduce6 : (!transform.any_op) -> !transform.any_op
  %fused6 = transform.air.fuse_multi_op_linalg %generic6, %reduce6_gen : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
