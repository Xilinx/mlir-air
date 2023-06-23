//===- air_custom.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @custom
// CHECK: air.custom operands ({{.*}}, {{.*}}) : i32, i32
func.func @custom(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.custom operands (%arg0, %arg0) : i32, i32 attributes { }
//   air.custom operands (%arg0, %arg0)
  return
}

// CHECK-LABEL: func.func @custom_async
// CHECK: %1 = air.custom async [%0] operands ({{.*}}, {{.*}}) : i32, i32
func.func @custom_async(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.custom async [%e0] operands (%arg0, %arg0) : i32, i32 attributes { }
  air.wait_all [%e0]
  return
}

// CHECK-LABEL: func.func @custom_emptyargs
// CHECK: %1 = air.custom async [%0]
func.func @custom_emptyargs(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.custom async [%e0] operands ()
  air.wait_all [%e0]
  return
}

// CHECK-LABEL: func.func @custom_noargs
// CHECK: %1 = air.custom async [%0]
func.func @custom_noargs(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.custom async [%e0]
  air.wait_all [%e0]
  return
}
