//===- air_transform_link_with.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// Transform dialect test: link_with attribute
// CHECK: transform.air.linalg_to_library_call

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    // Should use "my_linked_func" as the function name and "extern_func.o" as link_with
    %call = transform.air.linalg_to_library_call %matmul { function_name = "my_linked_func", link_with = "extern_func.o" } : (!pdl.operation) -> !pdl.operation
  }
}
