//===- air_transform_ops.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// This doesn't do much because this file is input to the
// air_transform_payload.mlir test

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.linalg_tile
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %matmul_1, %loop = transform.air.linalg_tile %matmul [64, 64, 0]
    transform.yield
  }
}
