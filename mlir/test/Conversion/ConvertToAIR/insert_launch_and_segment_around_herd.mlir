//===- insert_launch_and_segment_around_herd.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-insert-launch-and-segment-around-herd | FileCheck %s

// CHECK-LABEL: func.func @herd
// CHECK: air.launch
// CHECK: air.segment
// CHECK: air.herd @herd
func.func @herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd @herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}

// CHECK-LABEL: func.func @two_herds
// CHECK: air.launch
// CHECK: air.segment
// CHECK: air.herd @herd_0
// CHECK: air.launch
// CHECK: air.segment
// CHECK: air.herd @herd_1
func.func @two_herds(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd @herd_0 tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  air.herd @herd_1 tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}

// CHECK-LABEL: func.func @async_herd
// CHECK: %[[EVENT0:.*]] = air.launch
// CHECK: %[[EVENT1:.*]] = air.segment
// CHECK: %[[EVENT2:.*]] = air.herd @herd
func.func @async_herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %token = air.herd @herd async tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}
