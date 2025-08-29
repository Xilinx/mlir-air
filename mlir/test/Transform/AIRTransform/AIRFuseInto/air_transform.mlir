//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.fuse_into_containing_op

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %matmul_1, %loops_1 = transform.air.linalg_tile %matmul [32, 32, 0]
  %fill_1 = transform.air.fuse_into_containing_op %fill into %loops_1
}
