//===- scf_forall_to_herd_launch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @scf0() {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: air.herd @herd_0  tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C2]], {{.*}}=%[[C2]])
func.func @scf0()  {
  scf.forall (%x,%y) in (2, 2) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf1() {
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], {{.*}}) in ({{.*}}=%[[C4]], {{.*}}=%[[C1]])
func.func @scf1()  {
  scf.forall (%x) in (4) {
    %2 = arith.muli %x, %x : index
  }
  return
}
