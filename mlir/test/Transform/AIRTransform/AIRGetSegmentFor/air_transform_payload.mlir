//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @get_segment_for_op
func.func @get_segment_for_op(%arg0: i32, %arg1: i32) {
  // expected-remark @below {{found segment}}
  air.segment args (%arg2=%arg0, %arg3=%arg1) : i32, i32 {
    %c1 = arith.constant 1 : index
    air.herd tile (%x, %y) in (%sx=%c1, %sy=%c1) args (%op0=%arg2, %op1=%arg3) : i32, i32 attributes { } {
      %2 = arith.addi %op0, %op1 : i32
    }
  }
  return
}
