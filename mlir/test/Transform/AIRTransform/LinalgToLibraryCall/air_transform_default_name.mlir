//===- air_transform_default_name.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// Transform dialect test: fallback to library_call attribute
// CHECK: transform.air.linalg_to_library_call

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Should use the library_call attribute ("linalg_matmul_view4x4xf32_view4x4xf32_view4x4xf32")
    %call = transform.air.linalg_to_library_call %matmul : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
