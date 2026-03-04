//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.get_segment_for
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1: (!transform.any_op) -> !transform.any_op
    %1 = transform.air.get_segment_for %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %1, "found segment" : !transform.any_op
      transform.yield
  }
}
