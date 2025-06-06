//===- insert_launch_around_herd.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-insert-launch-around-herd | FileCheck %s
// RUN: air-opt %s -air-insert-launch-around-herd="insert-segment=true" | FileCheck %s  --check-prefix=INSERTSEG

// CHECK-LABEL: func.func @herd
// CHECK: air.launch
// CHECK: air.herd @herd
// INSERTSEG-LABEL: func.func @herd
// INSERTSEG: air.launch
// INSERTSEG: air.segment
// INSERTSEG: air.herd @herd
func.func @herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd @herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  return
}

// CHECK-LABEL: func.func @two_herds
// CHECK: air.launch
// CHECK: air.herd @herd_0
// CHECK: air.launch
// CHECK: air.herd @herd_1
// INSERTSEG-LABEL: func.func @two_herds
// INSERTSEG: air.launch
// INSERTSEG: air.segment
// INSERTSEG: air.herd @herd_0
// INSERTSEG: air.launch
// INSERTSEG: air.segment
// INSERTSEG: air.herd @herd_1
func.func @two_herds(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd @herd_0 tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  air.herd @herd_1 tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  return
}

// CHECK-LABEL: func.func @async_herd
// CHECK: %[[EVENT0:.*]] = air.launch
// CHECK: %[[EVENT2:.*]] = air.herd @herd
// INSERTSEG-LABEL: func.func @async_herd
// INSERTSEG: %[[EVENT0:.*]] = air.launch
// INSERTSEG: %[[EVENT1:.*]] = air.segment
// INSERTSEG: %[[EVENT2:.*]] = air.herd @herd
func.func @async_herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %token = air.herd @herd async tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  return
}
