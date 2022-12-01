//===- air_execute.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

func.func @air_execute_0() {

  // CHECK: air.execute -> (index)
  // CHECK: air.execute_terminator {{.*}} : index
  %1, %2 = air.execute -> (index) {
    %c1 = arith.constant 1 : index
    air.execute_terminator %c1 : index
  }

  %e = air.wait_all async
  // {{.*}} = air.execute [{{.*}}] {
  %3 = air.execute [%e] {} {foo = "bar"}

  // CHECK: air.execute -> (index)
  // CHECK: air.execute_terminator {{.*}} : index
  %4, %5 = air.execute -> (index) {
    %c1 = arith.constant 1 : index
    air.execute_terminator %c1 : index
  } {id = 1}

  // CHECK %{{.*}}, %{{.*}}:2 = air.execute [{{.*}}] -> (index, i32) {
  %6, %7, %8 = air.execute[%e, %4] -> (index, i32) {
    %c2 = arith.constant 2 : index
    %i2 = arith.constant 2 : i32
    air.execute_terminator %c2, %i2 : index, i32
  }
  return
}