//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.vector_type_cast

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Basic vector.fma operation cast from f32 to f16
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_fma_f32_to_f16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %fma1 = transform.structured.match ops{["vector.fma"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %result1 = transform.air.vector_type_cast %fma1 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op

  // Test case 2: Vector addition with bf16 target type
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_add_f32_to_bf16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %addf2 = transform.structured.match ops{["arith.addf"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %result2 = transform.air.vector_type_cast %addf2 {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

  // Test case 3: Vector multiplication with 2D vectors
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_mul_2d_f32_to_f16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %mulf3 = transform.structured.match ops{["arith.mulf"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %result3 = transform.air.vector_type_cast %mulf3 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op

  // Test case 4: Vector contract operation (matrix multiplication)
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_contract_f32_to_f16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %contract4 = transform.structured.match ops{["vector.contract"]} in %func4 : (!transform.any_op) -> !transform.any_op
  %result4 = transform.air.vector_type_cast %contract4 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op

  // Test case 5: Vector reduction operation
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_reduction_f32_to_f16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %reduction5 = transform.structured.match ops{["vector.reduction"]} in %func5 : (!transform.any_op) -> !transform.any_op
  %result5 = transform.air.vector_type_cast %reduction5 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op

  // Test case 7: Integer vector operations (i32 to i16)
  %func7 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_int_i32_to_i16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %addi7 = transform.structured.match ops{["arith.addi"]} in %func7 : (!transform.any_op) -> !transform.any_op
  %result7 = transform.air.vector_type_cast %addi7 {target_element_type = i16} : (!transform.any_op) -> !transform.any_op

  // Test case 8: Multiple vector operations in sequence
  %func8 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_sequence_f32_to_f16"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %addf8 = transform.structured.match ops{["arith.addf"]} in %func8 : (!transform.any_op) -> !transform.any_op
  %mulf8 = transform.structured.match ops{["arith.mulf"]} in %func8 : (!transform.any_op) -> !transform.any_op
  %result8a = transform.air.vector_type_cast %addf8 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op
  %result8b = transform.air.vector_type_cast %mulf8 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op

  // Test case 9: Single-element INPUT vector should NOT be cast (new feature)
  %func9 = transform.structured.match ops{["func.func"]} attributes{sym_name = "single_element_input_not_cast"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %addf9 = transform.structured.match ops{["arith.addf"]} in %func9 : (!transform.any_op) -> !transform.any_op
  %result9 = transform.air.vector_type_cast %addf9 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op

  // Test case 10: vector.multi_reduction with single-element output (new feature)
  %func10 = transform.structured.match ops{["func.func"]} attributes{sym_name = "multi_reduction_single_output"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %multi_red10 = transform.structured.match ops{["vector.multi_reduction"]} in %func10 : (!transform.any_op) -> !transform.any_op
  %result10 = transform.air.vector_type_cast %multi_red10 {target_element_type = f16} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
