//===- air_collapse_herd.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-collapse-herd -canonicalize --split-input-file | FileCheck %s

// CHECK: func.func @test0
// CHECK: %[[CST9:.*]] = arith.constant 9 : index
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: air.herd  tile (%[[VAL0:.*]], %[[VAL1:.*]]) in (%[[VAL2:.*]]=%[[CST1]], %[[VAL3:.*]]=%[[CST9]])

func.func @test0() -> () {
  %c3 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%c3, %sy=%c3) {
  }
  return
}

// -----

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 2 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 2 >= 0, s1 == 0)>
// CHECK: func.func @test1
// CHECK: %[[CST9:.*]] = arith.constant 9 : index
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: air.herd  tile (%[[ARG0:.*]], %[[ARG1:.*]]) in (%[[ARG2:.*]]=%[[CST1]], %[[ARG3:.*]]=%[[CST9]])
// CHECK:   %[[CST0:.*]] = arith.constant 0 : index
// CHECK:   %[[CST3:.*]] = arith.constant 3 : index
// CHECK:   %[[VAL0:.*]] = arith.remsi %[[ARG1]], %[[CST3]] : index
// CHECK:   %[[VAL1:.*]] = arith.divsi %[[ARG1]], %[[CST3]] : index
// CHECK:   affine.if [[$SET0]]()[%[[VAL1]], %[[VAL0]]] {
// CHECK:   } else {
// CHECK:   }
// CHECK:   affine.if [[$SET1]]()[%[[VAL1]], %[[VAL0]]] {
// CHECK:   } else {

#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 2 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 2 >= 0, s1 == 0)>
func.func @test1() -> () {
  %c3 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%c3, %sy=%c3) {
    %c0 = arith.constant 0 : index
    affine.if #set0()[%x, %y] {
      %alloc = memref.alloc() : memref<8x16xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<8x16xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<8x16xi32, 2>
      memref.dealloc %alloc : memref<8x16xi32, 2>
    } else {
      %alloc = memref.alloc() : memref<16x16xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<16x16xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<16x16xi32, 2>
      memref.dealloc %alloc : memref<16x16xi32, 2>
    }
    affine.if #set1()[%x, %y] {
      %alloc = memref.alloc() : memref<16x8xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<16x8xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<16x8xi32, 2>
      memref.dealloc %alloc : memref<16x8xi32, 2>
    } else {
      %alloc = memref.alloc() : memref<8x8xi32, 2>
      %2 = memref.load %alloc[%c0, %c0] : memref<8x8xi32, 2>
      memref.store %2, %alloc[%c0, %c0] : memref<8x8xi32, 2>
      memref.dealloc %alloc : memref<8x8xi32, 2>
    }
  }
  return
}
