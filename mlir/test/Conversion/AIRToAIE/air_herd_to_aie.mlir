//===- air_herd_to_aie.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie --split-input-file | FileCheck %s
module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  // CHECK-LABEL: aie.device
  // CHECK: %[[VAR1:.*]] = aie.tile(1, 1)
  // CHECK: %[[BUF1:.*]] = aie.buffer(%[[VAR1]]) {{{.*}}} : memref<1xi32, 2>
  // CHECK: %[[BUF2:.*]] = aie.buffer(%[[VAR1]]) {{{.*}}} : memref<1xi32, 2>
  // CHECK: %[[BUF3:.*]] = aie.buffer(%[[VAR1]]) {{{.*}}} : memref<1xi32, 2>
  // CHECK: %[[VAR2:.*]] = aie.core(%[[VAR1]])  {
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src0 = memref.alloc() : memref<1xi32, 2>
    %src1 = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    // CHECK: load %[[BUF1]]
    %0 = memref.load %src0[%zero] : memref<1xi32, 2>
    // CHECK: load %[[BUF2]]
    %1 = memref.load %src1[%zero] : memref<1xi32, 2>
    %2 = arith.addi %0, %1 :  i32
    %dst0 = memref.alloc() : memref<1xi32, 2>
    // CHECK: memref.store {{.*}}, %[[BUF3]]
    memref.store %2, %dst0[%zero] : memref<1xi32, 2>
  }
  // CHECK: sym_name = "herd_0"
  return
}

}

// -----

// Test that L1-to-L1 memref.copy is lowered to loops with load/store.
// CHECK: aie.device
// CHECK: %[[TILE:.*]] = aie.tile(1, 1)
// CHECK: %[[BUF1:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: %[[BUF0:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: aie.core(%[[TILE]]) {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.load %[[BUF1]]
// CHECK:       memref.store {{.*}}, %[[BUF0]]
module {

func.func @memref_copy_l1_to_l1() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src = memref.alloc() : memref<4x8xi32, 2>
    %dst = memref.alloc() : memref<4x8xi32, 2>
    memref.copy %src, %dst : memref<4x8xi32, 2> to memref<4x8xi32, 2>
    air.herd_terminator
  }
  return
}

}

// -----

// Test that L1-to-L1 memref.copy wrapped in air.execute is lowered to loops.
// CHECK: aie.device
// CHECK: %[[TILE:.*]] = aie.tile(1, 1)
// CHECK: %[[BUF1:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: %[[BUF0:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: aie.core(%[[TILE]]) {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.load %[[BUF1]]
// CHECK:       memref.store {{.*}}, %[[BUF0]]
module {

func.func @memref_copy_l1_to_l1_in_execute() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src = memref.alloc() : memref<4x8xi32, 2>
    %dst = memref.alloc() : memref<4x8xi32, 2>
    %0 = air.execute {
      memref.copy %src, %dst : memref<4x8xi32, 2> to memref<4x8xi32, 2>
      air.execute_terminator
    }
    air.herd_terminator
  }
  return
}

}

// -----

// Test that L1-to-L1 linalg.copy is lowered to loops with load/store.
// CHECK: aie.device
// CHECK: %[[TILE:.*]] = aie.tile(1, 1)
// CHECK: %[[BUF1:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: %[[BUF0:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: aie.core(%[[TILE]]) {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.load %[[BUF1]]
// CHECK:       memref.store {{.*}}, %[[BUF0]]
module {

func.func @linalg_copy_l1_to_l1() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src = memref.alloc() : memref<4x8xi32, 2>
    %dst = memref.alloc() : memref<4x8xi32, 2>
    linalg.copy ins(%src : memref<4x8xi32, 2>) outs(%dst : memref<4x8xi32, 2>)
    air.herd_terminator
  }
  return
}

}

// -----

// Test that L1-to-L1 linalg.copy wrapped in air.execute is lowered to loops.
// CHECK: aie.device
// CHECK: %[[TILE:.*]] = aie.tile(1, 1)
// CHECK: %[[BUF1:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: %[[BUF0:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<4x8xi32, 2>
// CHECK: aie.core(%[[TILE]]) {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.load %[[BUF1]]
// CHECK:       memref.store {{.*}}, %[[BUF0]]
module {

func.func @linalg_copy_l1_to_l1_in_execute() {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src = memref.alloc() : memref<4x8xi32, 2>
    %dst = memref.alloc() : memref<4x8xi32, 2>
    %0 = air.execute {
      linalg.copy ins(%src : memref<4x8xi32, 2>) outs(%dst : memref<4x8xi32, 2>)
      air.execute_terminator
    }
    air.herd_terminator
  }
  return
}

}
