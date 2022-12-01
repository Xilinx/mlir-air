//===- air_herd_launch.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @launch
// CHECK: air.herd tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}})
func.func @launch(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}

// CHECK-LABEL: func.func @launch_async
// CHECK: %1 = air.herd async [%0]
func.func @launch_async(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  air.wait_all [%e0]
  return
}

// CHECK-LABEL: func.func @launch_emptyargs
// CHECK: %1 = air.herd async [%0] tile ({{.*}}) in ({{.*}}) {
func.func @launch_emptyargs(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args () {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    air.herd_terminator
  }
  air.wait_all [%e0]
  return
}

// CHECK-LABEL: func.func @launch_noargs
// CHECK: %1 = air.herd async [%0] tile ({{.*}}) in ({{.*}}) {
func.func @launch_noargs(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    air.herd_terminator
  }
  air.wait_all [%e0]
  return
}
