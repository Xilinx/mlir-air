//===- command_line.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds="num-rows=16 num-cols=32 row-anchor=4 col-anchor=6" | FileCheck %s

// CHECK-LABEL: test_partition_cl
// CHECK: air.partition
// CHECK-SAME: attributes {x_loc = 6 : i64, x_size = 32 : i64, y_loc = 4 : i64, y_size = 16 : i64}
// CHECK: air.herd
// CHECK-SAME: attributes {x_loc = 6 : i64, y_loc = 4 : i64}
func.func @test_partition_cl() -> () {
  air.partition {
    %c2 = arith.constant 2 : index
    air.herd tile (%x, %y) in (%sx=%c2, %sy=%c2)  {
    }
  }
  return
}

// CHECK-LABEL: test_partition_attr
// CHECK: air.partition
// CHECK-SAME: attributes {x_loc = 0 : i64, x_size = 10 : i64, y_loc = 0 : i64, y_size = 10 : i64}
// CHECK: air.herd
// CHECK-SAME: attributes {x_loc = 0 : i64, y_loc = 0 : i64}
func.func @test_partition_attr() -> () {
  air.partition attributes {x_loc = 0 : i64, x_size = 10 : i64, y_loc = 0 : i64, y_size = 10 : i64} {
    %c4 = arith.constant 4 : index
    air.herd tile (%x, %y) in (%sx=%c4, %sy=%c4) {
    }
  }
  return
}

// CHECK-LABEL: test_herd_cl
// CHECK: air.herd
// CHECK-SAME: attributes {x_loc = 6 : i64, y_loc = 4 : i64}
func.func @test_herd_cl() -> () {
  %c3 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%c3, %sy=%c3) {
  }
  return
}

// CHECK-LABEL: test_herd_attr
// CHECK: air.herd
// CHECK-SAME: attributes {x_loc = 2 : i64, y_loc = 3 : i64}
func.func @test_herd_attr() -> () {
  %c3 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%c3, %sy=%c3) attributes {x_loc=2 : i64, y_loc = 3 : i64} {
  }
  return
}