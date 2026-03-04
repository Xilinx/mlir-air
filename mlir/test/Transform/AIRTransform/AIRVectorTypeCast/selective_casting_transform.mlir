//===- selective_casting_transform.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.vector_type_cast

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Default behavior - cast all (empty indices)
  // This demonstrates backward compatibility
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "contract_cast_all_default"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %contract1 = transform.structured.match ops{["vector.contract"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %result1 = transform.air.vector_type_cast %contract1 {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

  // Test case 2: Cast lhs and rhs (inputs 0, 1); output cast by default so acc also needs to match
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "contract_cast_lhs_rhs_with_acc_output"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %contract2 = transform.structured.match ops{["vector.contract"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %result2 = transform.air.vector_type_cast %contract2 {target_element_type = bf16, input_indices = [0, 1]} : (!transform.any_op) -> !transform.any_op

  // Test case 3: Cast only acc and result (input 2, output 0) - respects vector.contract type constraints
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "contract_cast_acc_result_only"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %contract3 = transform.structured.match ops{["vector.contract"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %result3 = transform.air.vector_type_cast %contract3 {target_element_type = bf16, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
