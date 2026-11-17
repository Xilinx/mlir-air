//===- segment_unroll_dynamic_size.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics -air-to-aie='device=npu2'

// Segment unroll sizes must be constants.
func.func @dynamic_x(%n: index) {
  air.launch () in () args(%a=%n) : index {
    // expected-error @below {{segment unroll X-dimension size must be a static constant}}
    air.segment @seg unroll (%x) in (%sx=%a) {
    }
  }
  return
}

// -----

func.func @dynamic_y(%n: index) {
  air.launch () in () args(%a=%n) : index {
    %c1 = arith.constant 1 : index
    // expected-error @below {{segment unroll Y-dimension size must be a static constant}}
    air.segment @seg unroll (%x, %y) in (%sx=%c1, %sy=%a) {
    }
  }
  return
}
