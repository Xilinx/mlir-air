//===- air_transform_default_name.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// Transform dialect test: fallback to library_call attribute
// CHECK: transform.air.linalg_to_library_call

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    // Should use the library_call attribute ("linalg_matmul_view4x4xf32_view4x4xf32_view4x4xf32")
    %call = transform.air.linalg_to_library_call %matmul : (!pdl.operation) -> !pdl.operation
  }
}
