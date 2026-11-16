//===- affine_par_to_herd_launch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @par0
// CHECK: %[[C0:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0 tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C0]], {{.*}}=%[[C0]])
func.func @par0()  {
  affine.parallel (%x,%y) = (0,0) to (1,1) {
    %2 = arith.addi %x, %y : index
    affine.yield
  }
  return
}

// -----

func.func @par1()  {
  // expected-error@+2 {{'affine.parallel' op failed conversion to 'air.herd': only 2d loops are supported}}
  // expected-error@+1 {{failed to legalize}}
  affine.parallel (%x,%y,%z) = (0,0,0) to (1,2,3) {
    %2 = arith.addi %x, %y : index
    affine.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @par2
func.func @par2()  {
  // CHECK: %[[C0:.*]] = arith.constant 4 : index
  // CHECK: %[[C1:.*]] = arith.constant 5 : index
  // CHECK: air.herd @herd_0 tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C0]], {{.*}}=%[[C1]])
  affine.parallel (%x,%y) = (0,2) to (4,12) step (1,2) {
    %2 = arith.addi %x, %y : index
    affine.yield
  }
  return
}

// -----

// Non-constant values used in the body become herd arguments; constants are
// cloned into the body instead.
// CHECK-LABEL: func.func @par3
// CHECK-SAME: %[[N:.*]]: index
// CHECK: air.herd @herd_0 {{.*}}tile (%[[X:.*]], %[[Y:.*]]) in ({{.*}}) args(%[[K:.*]]=%[[N]]) : index {
// CHECK: %[[C7:.*]] = arith.constant 7 : index
// CHECK: %[[A:.*]] = arith.addi %[[X]], %[[K]] : index
// CHECK: arith.muli %[[A]], %[[C7]] : index
func.func @par3(%n: index)  {
  %c7 = arith.constant 7 : index
  affine.parallel (%x,%y) = (0,0) to (4,4) {
    %b = memref.alloc() : memref<4xindex, 2 : i32>
    %2 = arith.addi %x, %n : index
    %3 = arith.muli %2, %c7 : index
    memref.store %3, %b[%y] : memref<4xindex, 2 : i32>
    affine.yield
  }
  return
}
