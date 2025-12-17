//===- air_transform.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.convert_size1_vector_to_scalar

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %transformed = transform.air.convert_size1_vector_to_scalar %func_op
}
