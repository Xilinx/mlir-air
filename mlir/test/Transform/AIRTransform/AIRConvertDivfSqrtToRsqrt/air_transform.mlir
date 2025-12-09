//===- air_transform.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.convert_divf_sqrt_to_rsqrt

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.air.convert_divf_sqrt_to_rsqrt %func
}
