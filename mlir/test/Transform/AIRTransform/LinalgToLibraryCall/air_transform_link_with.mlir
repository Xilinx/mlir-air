//===- air_transform_link_with.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// Transform dialect test: link_with attribute
// CHECK: transform.air.linalg_to_library_call

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Should use "my_linked_func" as the function name and "extern_func.o" as link_with
    %call = transform.air.linalg_to_library_call %matmul { function_name = "my_linked_func", link_with = "extern_func.o" } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
