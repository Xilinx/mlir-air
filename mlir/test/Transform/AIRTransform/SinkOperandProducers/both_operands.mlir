//===- both_operands.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_both.mlir' %s | FileCheck %s

// Both operands' producers move past the store; the scf.if takes the value
// its region uses along.
// CHECK-LABEL: @region_capture
// CHECK: memref.store
// CHECK: %[[V:.*]] = arith.addf %arg0, %arg1
// CHECK: %[[X:.*]] = scf.if
// CHECK: scf.yield %[[V]]
// CHECK: %[[N:.*]] = arith.negf %arg1
// CHECK: arith.mulf %[[X]], %[[N]] {sink}
func.func @region_capture(%a: f32, %b: f32, %c: i1, %m: memref<f32>) -> f32 {
  %v = arith.addf %a, %b : f32
  %x = scf.if %c -> f32 {
    scf.yield %v : f32
  } else {
    scf.yield %a : f32
  }
  %n = arith.negf %b : f32
  memref.store %a, %m[] : memref<f32>
  %r = arith.mulf %x, %n {sink} : f32
  return %r : f32
}

// The scf.if stores: it stays, and so does the value its region uses, which
// the target also reads.
// CHECK-LABEL: @kept_region_use
// CHECK: %[[V:.*]] = arith.addf %arg0, %arg1
// CHECK: %[[X:.*]] = scf.if
// CHECK: memref.store %[[V]]
// CHECK: memref.store %arg0
// CHECK: arith.mulf %[[X]], %[[V]] {sink}
func.func @kept_region_use(%a: f32, %b: f32, %c: i1, %m: memref<f32>) -> f32 {
  %v = arith.addf %a, %b : f32
  %x = scf.if %c -> f32 {
    memref.store %v, %m[] : memref<f32>
    scf.yield %a : f32
  } else {
    scf.yield %a : f32
  }
  memref.store %a, %m[] : memref<f32>
  %r = arith.mulf %x, %v {sink} : f32
  return %r : f32
}

// Only producers in the target's block move: the one outside the loop stays.
// CHECK-LABEL: @outer
// CHECK: %[[O:.*]] = arith.negf %arg0
// CHECK: scf.for
// CHECK: memref.store
// CHECK: %[[I:.*]] = arith.addf %arg0, %arg0
// CHECK: arith.mulf %[[O]], %[[I]] {sink}
func.func @outer(%a: f32, %m: memref<f32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %o = arith.negf %a : f32
  scf.for %k = %c0 to %n step %c1 {
    %in = arith.addf %a, %a : f32
    memref.store %a, %m[] : memref<f32>
    %r = arith.mulf %o, %in {sink} : f32
    memref.store %r, %m[] : memref<f32>
  }
  return
}

// Two targets, the first an operand of the second: each moves its own
// producers.
// CHECK-LABEL: @two_targets
// CHECK: memref.store %arg0
// CHECK: %[[P:.*]] = arith.negf %arg0
// CHECK: %[[T2:.*]] = arith.addf %[[P]], %arg1 {sink}
// CHECK: %[[Q:.*]] = arith.negf %arg1
// CHECK: arith.addf %[[T2]], %[[Q]] {sink}
func.func @two_targets(%a: f32, %b: f32, %m: memref<f32>) -> f32 {
  %p = arith.negf %a : f32
  memref.store %a, %m[] : memref<f32>
  %t2 = arith.addf %p, %b {sink} : f32
  %q = arith.negf %b : f32
  memref.store %b, %m[] : memref<f32>
  %t1 = arith.addf %t2, %q {sink} : f32
  return %t1 : f32
}
