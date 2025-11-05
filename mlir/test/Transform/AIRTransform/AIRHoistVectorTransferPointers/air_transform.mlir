//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_vector_transfer_pointers

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // Test case 1: Basic hoisting with 2D memref - loop-invariant indices
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_2d_transfers"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %hoisted1 = transform.air.hoist_vector_transfer_pointers %loop1

  // Test case 2: Hoisting with loop IV-dependent indices
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_iv_dependent_indices"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!pdl.operation) -> !pdl.operation
  %hoisted2 = transform.air.hoist_vector_transfer_pointers %loop2
  
  // Test case 3: IV in higher dimension - tests correct stride calculation
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_iv_in_higher_dimension"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!pdl.operation) -> !pdl.operation
  %hoisted3 = transform.air.hoist_vector_transfer_pointers %loop3
}
