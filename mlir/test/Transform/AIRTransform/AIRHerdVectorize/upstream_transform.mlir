//===- upstream_transform.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.structured.vectorize_children_and_apply_patterns

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %herd_op = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herd = transform.structured.vectorize_children_and_apply_patterns %herd_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
