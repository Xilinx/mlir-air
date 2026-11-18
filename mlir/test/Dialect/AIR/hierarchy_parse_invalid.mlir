//===- hierarchy_parse_invalid.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics

func.func @fewer_sizes_than_ids() {
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{expected one size per id}}
  air.herd tile (%x, %y) in (%sx = %c1) {
  }
  return
}

// -----

func.func @fewer_types_than_args(%arg0: memref<4xi32>, %arg1: memref<4xi32>) {
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{expected one type per kernel operand}}
  air.launch (%x) in (%sx = %c1) args(%a = %arg0, %b = %arg1) : memref<4xi32> {
  }
  return
}
