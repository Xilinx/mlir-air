//===- simple_transform.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: transform.air.hoist_cast_pair

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %func1 = transform.structured.match ops{["func.func"]} attributes{sym_name = "hoist_simple_no_shapecast"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop1 = transform.structured.match ops{["scf.for"]} in %func1 : (!pdl.operation) -> !pdl.operation
  %extsi1 = transform.structured.match ops{["arith.extsi"]} in %loop1 : (!pdl.operation) -> !pdl.operation
  %trunci1 = transform.structured.match ops{["arith.trunci"]} in %loop1 : (!pdl.operation) -> !pdl.operation
  %new_loop1 = transform.air.hoist_cast_pair %extsi1, %trunci1, %loop1
}
