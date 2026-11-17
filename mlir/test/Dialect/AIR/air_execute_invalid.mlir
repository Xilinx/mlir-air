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
