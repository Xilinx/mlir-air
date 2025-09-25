//===- air_transform_explicit_name.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// Transform dialect test: explicit function_name
// CHECK: transform.air.linalg_to_library_call

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    // Should use "my_explicit_func" as the function name
    %call = transform.air.linalg_to_library_call %matmul { function_name = "my_explicit_func" } : (!pdl.operation) -> !pdl.operation
  }
}
