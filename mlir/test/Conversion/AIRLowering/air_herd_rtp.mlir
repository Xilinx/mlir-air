//===- air_herd_rtp.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s
// CHECK-LABEL: func.func @herd1
// CHECK: %{{.*}} = airrt.herd_load "herd" (%{{.*}}, %{{.*}}) : (i32, i32) -> i64
func.func @herd1(%arg0: i32, %arg1: i32) {
  %cst1 = arith.constant 1 : index
  %cst2 = arith.constant 2 : index
  air.herd @herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst2) args(%a = %arg0, %b = %arg1) : i32, i32 {
    %src0 = memref.alloc() : memref<1xi32, 2>
    %src1 = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    %0 = memref.load %src0[%zero] : memref<1xi32, 2>
    %1 = memref.load %src1[%zero] : memref<1xi32, 2>
    %2 = arith.addi %0, %a : i32
    %3 = arith.addi %1, %b : i32
    %4 = arith.addi %2, %3 : i32
    %dst0 = memref.alloc() : memref<1xi32, 2>
    memref.store %4, %dst0[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}

