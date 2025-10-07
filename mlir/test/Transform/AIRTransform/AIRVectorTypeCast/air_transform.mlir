//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.vector_type_cast

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic vector.fma operation cast from f32 to f16
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_fma_f32_to_f16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %fma1 = transform.structured.match ops{["vector.fma"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %result1 = transform.air.vector_type_cast %fma1 {target_element_type = f16}

  // Test case 2: Vector addition with bf16 target type
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_add_f32_to_bf16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %addf2 = transform.structured.match ops{["arith.addf"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %result2 = transform.air.vector_type_cast %addf2 {target_element_type = bf16}

  // Test case 3: Vector multiplication with 2D vectors
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_mul_2d_f32_to_f16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %mulf3 = transform.structured.match ops{["arith.mulf"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %result3 = transform.air.vector_type_cast %mulf3 {target_element_type = f16}

  // Test case 4: Vector contract operation (matrix multiplication)
  %func4 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_contract_f32_to_f16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %contract4 = transform.structured.match ops{["vector.contract"]} in %func4 : (!pdl.operation) -> !pdl.operation
  %result4 = transform.air.vector_type_cast %contract4 {target_element_type = f16}

  // Test case 5: Vector reduction operation
  %func5 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_reduction_f32_to_f16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %reduction5 = transform.structured.match ops{["vector.reduction"]} in %func5 : (!pdl.operation) -> !pdl.operation
  %result5 = transform.air.vector_type_cast %reduction5 {target_element_type = f16}

  // Test case 7: Integer vector operations (i32 to i16)
  %func7 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_int_i32_to_i16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %addi7 = transform.structured.match ops{["arith.addi"]} in %func7 : (!pdl.operation) -> !pdl.operation
  %result7 = transform.air.vector_type_cast %addi7 {target_element_type = i16}

  // Test case 8: Multiple vector operations in sequence
  %func8 = transform.structured.match ops{["func.func"]} attributes{sym_name = "vector_sequence_f32_to_f16"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %addf8 = transform.structured.match ops{["arith.addf"]} in %func8 : (!pdl.operation) -> !pdl.operation
  %mulf8 = transform.structured.match ops{["arith.mulf"]} in %func8 : (!pdl.operation) -> !pdl.operation
  %result8a = transform.air.vector_type_cast %addf8 {target_element_type = f16}
  %result8b = transform.air.vector_type_cast %mulf8 {target_element_type = f16}
}
