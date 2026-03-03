//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_vector_transfer_pointers

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  // Test case 1: Basic hoisting with 2D memref - loop-invariant indices
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_2d_transfers"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %hoisted1 = transform.air.hoist_vector_transfer_pointers %loop1 : (!transform.any_op) -> !transform.any_op

  // Test case 2: Hoisting with loop IV-dependent indices
  %func2 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_with_iv_dependent_indices"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop2 = transform.structured.match ops{["scf.for"]} in %func2 : (!transform.any_op) -> !transform.any_op
  %hoisted2 = transform.air.hoist_vector_transfer_pointers %loop2 : (!transform.any_op) -> !transform.any_op
  
  // Test case 3: IV in higher dimension - tests correct stride calculation
  %func3 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_iv_in_higher_dimension"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop3 = transform.structured.match ops{["scf.for"]} in %func3 : (!transform.any_op) -> !transform.any_op
  %hoisted3 = transform.air.hoist_vector_transfer_pointers %loop3 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
