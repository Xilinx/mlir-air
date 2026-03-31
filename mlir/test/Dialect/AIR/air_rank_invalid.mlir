//===- air_rank_invalid.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics

// Test: rank cannot be nested inside launch
func.func @rank_in_launch() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls = %c1) {
    %c1_0 = arith.constant 1 : index
    // expected-error @below {{must be the outermost hierarchy op}}
    air.rank (%rx) in (%sx = %c1_0) {
    }
  }
  return
}

// -----

// Test: rank cannot be nested inside segment
func.func @rank_in_segment() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls = %c1) {
    air.segment @seg0 {
      %c1_1 = arith.constant 1 : index
      // expected-error @below {{must be the outermost hierarchy op}}
      air.rank (%rx) in (%rsx = %c1_1) {
      }
    }
  }
  return
}

// -----

// Test: rank cannot be nested inside another rank
func.func @rank_in_rank() {
  %c1 = arith.constant 1 : index
  air.rank (%rx) in (%rsx = %c1) {
    %c1_0 = arith.constant 1 : index
    // expected-error @below {{must be the outermost hierarchy op}}
    air.rank (%rx2) in (%rsx2 = %c1_0) {
    }
  }
  return
}
