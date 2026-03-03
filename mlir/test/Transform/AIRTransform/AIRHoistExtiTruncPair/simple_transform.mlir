//===- simple_transform.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_cast_pair

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_no_shapecast"} in %arg1 : (!transform.any_op) -> !transform.any_op
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!transform.any_op) -> !transform.any_op
  %extsi1 = transform.structured.match ops{["arith.extsi"]} in %loop1 : (!transform.any_op) -> !transform.any_op
  %trunci1 = transform.structured.match ops{["arith.trunci"]} in %loop1 : (!transform.any_op) -> !transform.any_op
  %new_loop1 = transform.air.hoist_cast_pair %extsi1, %trunci1, %loop1 : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
