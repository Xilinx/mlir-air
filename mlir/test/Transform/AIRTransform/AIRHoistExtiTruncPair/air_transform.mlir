//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_cast_pair

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic hoisting of single extsi/trunci pair
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_extsi_trunci"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %extsi1 = transform.structured.match ops{["arith.extsi"]} in %loop1 : (!pdl.operation) -> !pdl.operation
  %trunci1 = transform.structured.match ops{["arith.trunci"]} in %loop1 : (!pdl.operation) -> !pdl.operation
  %new_loop1 = transform.air.hoist_cast_pair %extsi1, %trunci1, %loop1

  // Test case 2: Hoisting with vector.shape_cast operations
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_shape_cast"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %extsi2 = transform.structured.match ops{["arith.extsi"]} in %loop2 : (!pdl.operation) -> !pdl.operation
  %trunci2 = transform.structured.match ops{["arith.trunci"]} in %loop2 : (!pdl.operation) -> !pdl.operation
  %new_loop2 = transform.air.hoist_cast_pair %extsi2, %trunci2, %loop2

  // Test case 3: Hoisting multiple pairs from the same loop
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_four_pairs"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop3_0 = transform.structured.match ops{["scf.for"]} in %func3 : (!pdl.operation) -> !pdl.operation
  
  // Hoist first pair
  %all_extsi_1 = transform.structured.match ops{["arith.extsi"]} in %loop3_0 : (!pdl.operation) -> !pdl.operation
  %all_trunci_1 = transform.structured.match ops{["arith.trunci"]} in %loop3_0 : (!pdl.operation) -> !pdl.operation
  %extsi3_1, %rest_extsi_1_1, %rest_extsi_1_2, %rest_extsi_1_3 = transform.split_handle %all_extsi_1 {num_result_handles = 4} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %trunci3_1, %rest_trunci_1_1, %rest_trunci_1_2, %rest_trunci_1_3 = transform.split_handle %all_trunci_1 {num_result_handles = 4} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %loop3_1 = transform.air.hoist_cast_pair %extsi3_1, %trunci3_1, %loop3_0
  
  // Hoist second pair - re-match from the updated loop
  %all_extsi_2 = transform.structured.match ops{["arith.extsi"]} in %loop3_1 : (!pdl.operation) -> !pdl.operation
  %all_trunci_2 = transform.structured.match ops{["arith.trunci"]} in %loop3_1 : (!pdl.operation) -> !pdl.operation
  %extsi3_2, %rest_extsi_2_1, %rest_extsi_2_2 = transform.split_handle %all_extsi_2 {num_result_handles = 3} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  %trunci3_2, %rest_trunci_2_1, %rest_trunci_2_2 = transform.split_handle %all_trunci_2 {num_result_handles = 3} : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  %loop3_2 = transform.air.hoist_cast_pair %extsi3_2, %trunci3_2, %loop3_1
  
  // Hoist third pair - re-match from the updated loop
  %all_extsi_3 = transform.structured.match ops{["arith.extsi"]} in %loop3_2 : (!pdl.operation) -> !pdl.operation
  %all_trunci_3 = transform.structured.match ops{["arith.trunci"]} in %loop3_2 : (!pdl.operation) -> !pdl.operation
  %extsi3_3, %rest_extsi_3_1 = transform.split_handle %all_extsi_3 {num_result_handles = 2} : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %trunci3_3, %rest_trunci_3_1 = transform.split_handle %all_trunci_3 {num_result_handles = 2} : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %loop3_3 = transform.air.hoist_cast_pair %extsi3_3, %trunci3_3, %loop3_2
  
  // Hoist fourth pair - re-match from the updated loop
  %all_extsi_4 = transform.structured.match ops{["arith.extsi"]} in %loop3_3 : (!pdl.operation) -> !pdl.operation
  %all_trunci_4 = transform.structured.match ops{["arith.trunci"]} in %loop3_3 : (!pdl.operation) -> !pdl.operation
  %loop3_final = transform.air.hoist_cast_pair %all_extsi_4, %all_trunci_4, %loop3_3
}
