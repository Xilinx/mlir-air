//===- air_execute_invalid.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics

// The region of an air.execute holds exactly one block.

func.func @no_block() {
  // expected-error @+1 {{region #0 ('region') failed to verify constraint: region with 1 blocks}}
  %t = "air.execute"() ({
  }) : () -> !air.async.token
  return
}

// -----

// The values an air.execute returns after its token are those its terminator
// yields, in number and type.

func.func @wrong_type() {
  // expected-error @+1 {{result #1 has type 'index', but its terminator yields 'i32'}}
  %t, %v = air.execute -> (index) {
    %c = arith.constant 1 : i32
    air.execute_terminator %c : i32
  }
  return
}

// -----

func.func @too_few() {
  // expected-error @+1 {{returns 1 value(s) after its token, but its terminator yields 0}}
  %t, %v = air.execute -> (index) {
    air.execute_terminator
  }
  return
}

// -----

func.func @too_many() {
  // expected-error @+1 {{returns 0 value(s) after its token, but its terminator yields 1}}
  %t = air.execute {
    %c = arith.constant 1 : index
    air.execute_terminator %c : index
  }
  return
}
