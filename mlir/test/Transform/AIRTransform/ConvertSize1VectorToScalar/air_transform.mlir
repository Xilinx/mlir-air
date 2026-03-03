//===- air_transform.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.convert_size1_vector_to_scalar

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %transformed = transform.air.convert_size1_vector_to_scalar %func_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
