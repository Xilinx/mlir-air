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
transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %matmul_1, %loops:2 = transform.air.linalg_tile %matmul [64, 64, 0]
}
