//===- air_transform_explicit_name.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// Transform dialect test: explicit function_name
// CHECK: transform.air.linalg_to_library_call

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Should use "my_explicit_func" as the function name
    %call = transform.air.linalg_to_library_call %matmul { function_name = "my_explicit_func" } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
