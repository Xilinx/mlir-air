//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.sink_herd_operand_views

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %h = transform.structured.match ops{["air.herd"]} attributes{epilogue_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.sink_herd_operand_views %h : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
