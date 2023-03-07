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
  // expected-error@+1 {{'affine.parallel' op failed conversion to 'air.herd': only 2d loops are supported}}
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
