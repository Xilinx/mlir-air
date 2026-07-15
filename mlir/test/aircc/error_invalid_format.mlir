//===- error_invalid_format.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that aircc rejects --output-format=elf on npu1 targets.

// RUN: not aircc %s --device=npu1 --output-format=elf --tmpdir=%t 2>&1 | FileCheck %s

// CHECK: Error: --output-format=elf is not supported for --device=npu1

module {
  func.func @copy(%arg0: memref<1024xui8>, %arg1: memref<1024xui8>) {
    return
  }
}
