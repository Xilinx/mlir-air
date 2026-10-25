//===- error_non_npu_device.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that aircc rejects a missing or non-NPU --device.

// RUN: not aircc %s --device=xcvc1902 --tmpdir=%t 2>&1 | FileCheck %s --check-prefix=VERSAL
// RUN: not aircc %s --tmpdir=%t 2>&1 | FileCheck %s --check-prefix=NONE

// VERSAL: Error: --device must name an NPU device (e.g. npu1, npu2), got 'xcvc1902'
// NONE: Error: --device must name an NPU device (e.g. npu1, npu2), got ''

module {
  func.func @copy(%arg0: memref<1024xui8>, %arg1: memref<1024xui8>) {
    return
  }
}
