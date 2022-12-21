//===- scf_parallel_to_herd_launch.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @scf0() {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: air.herd @herd_0  tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C2]], {{.*}}=%[[C2]])
func.func @scf0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// -----

func.func @scferror0(%c0 : index)  {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: lower bound is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// -----

func.func @scferror1(%c1 : index)  {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: step is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// -----

func.func @scferror2(%c2 : index)  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: upper bound is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// -----

func.func @scferror3()  {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: step '2' does not evenly divide range '7'}}
  scf.parallel (%x,%y) = (%c2, %c2) to (%c9, %c9) step (%c2, %c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// -----

// CHECK: #[[M0:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: func.func @scf1() {
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], {{.*}}) in ({{.*}}=%[[C4]], {{.*}}=%[[C1]])
// CHECK: affine.apply #[[M0]](%[[A0]])
func.func @scf1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %2 = arith.muli %x, %x : index
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf2() {
// CHECK: %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK: scf.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_0]], %[[VAL_2]]) step (%[[VAL_0]], %[[VAL_0]]) {
// CHECK:   %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:   %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:   air.herd @herd_0  tile (%[[VAL_7:.*]], %[[VAL_8:.*]]) in (%[[VAL_9:.*]]=%[[VAL_5]], %[[VAL_10:.*]]=%[[VAL_6]]) args(%[[VAL_11:.*]]=%[[VAL_3]], %[[VAL_12:.*]]=%[[VAL_4]]) : index, index
func.func @scf2()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%a,%b,%x,%y) = (%c0,%c0,%c0,%c0) to (%c1,%c2,%c3,%c4) step (%c1,%c1,%c1,%c1) {
    %2 = arith.muli %x, %y : index
    scf.yield
  }
  return
}
