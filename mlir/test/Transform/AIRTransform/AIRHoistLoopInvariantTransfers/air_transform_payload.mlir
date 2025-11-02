//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic hoisting of loop-invariant vector transfers
// CHECK-LABEL: @hoist_simple_loop_invariant
func.func @hoist_simple_loop_invariant(%arg0: memref<16x16xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  
  // CHECK: %[[READ:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK-NEXT: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[READ]])
  scf.for %i = %c0 to %c4 step %c1 {
    %val = vector.transfer_read %arg0[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<16x16xi32, 2>, vector<4x4xi32>
    // Some computation
    %result = arith.addi %val, %val : vector<4x4xi32>
    vector.transfer_write %result, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xi32>, memref<16x16xi32, 2>
    // CHECK: %[[COMPUTE:.*]] = arith.addi %[[ITER]], %[[ITER]]
    // CHECK-NEXT: scf.yield %[[COMPUTE]]
  }
  // CHECK: vector.transfer_write %[[LOOP]], %{{.*}}[%{{.*}}, %{{.*}}]
  return
}

// Test case 2: Hoisting with affine indices
// CHECK-LABEL: @hoist_with_affine_indices
#map = affine_map<()[s0] -> (s0 * 4)>
func.func @hoist_with_affine_indices(%arg0: memref<32x32xi16, 2>, %x: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i16 = arith.constant 0 : i16
  
  %idx = affine.apply #map()[%x]
  
  // CHECK: %[[AFFINE:.*]] = affine.apply
  // CHECK: %[[READ:.*]] = vector.transfer_read %{{.*}}[{{.*}}]
  // CHECK-NEXT: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[READ]])
  scf.for %i = %c0 to %c8 step %c1 {
    %val = vector.transfer_read %arg0[%idx, %c0], %c0_i16 {in_bounds = [true, true]} : memref<32x32xi16, 2>, vector<8x8xi16>
    %result = arith.addi %val, %val : vector<8x8xi16>
    vector.transfer_write %result, %arg0[%idx, %c0] {in_bounds = [true, true]} : vector<8x8xi16>, memref<32x32xi16, 2>
    // CHECK: %[[COMPUTE:.*]] = arith.addi %[[ITER]], %[[ITER]]
    // CHECK-NEXT: scf.yield %[[COMPUTE]]
  }
  // CHECK: vector.transfer_write %[[LOOP]], %{{.*}}[%[[AFFINE]], %{{.*}}]
  return
}

// Test case 3: Hoisting two pairs from the same loop (tests handle chaining)
// CHECK-LABEL: @hoist_two_pairs_from_same_loop
#map1 = affine_map<()[s0] -> (s0 + 1)>
func.func @hoist_two_pairs_from_same_loop(%arg0: memref<32x32xi32, 2>, %x: index, %y: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  
  // Compute affine index outside the loop (loop-invariant)
  %x_plus_1 = affine.apply #map1()[%x]
  
  // CHECK: %[[READ1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: %[[AFFINE2:.*]] = affine.apply
  // CHECK: %[[READ2:.*]] = vector.transfer_read %{{.*}}[%[[AFFINE2]], %{{.*}}]
  // CHECK-NEXT: %[[LOOP:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER1:.*]] = %[[READ1]], %[[ITER2:.*]] = %[[READ2]])
  scf.for %i = %c0 to %c4 step %c1 {
    // First pair: read/write at [x, y]
    %val1 = vector.transfer_read %arg0[%x, %y], %c0_i32 {in_bounds = [true, true]} : memref<32x32xi32, 2>, vector<4x4xi32>
    %result1 = arith.addi %val1, %val1 : vector<4x4xi32>
    
    // Second pair: read/write at [x+1, y]
    %val2 = vector.transfer_read %arg0[%x_plus_1, %y], %c0_i32 {in_bounds = [true, true]} : memref<32x32xi32, 2>, vector<4x4xi32>
    %result2 = arith.muli %val2, %val2 : vector<4x4xi32>
    
    vector.transfer_write %result1, %arg0[%x, %y] {in_bounds = [true, true]} : vector<4x4xi32>, memref<32x32xi32, 2>
    vector.transfer_write %result2, %arg0[%x_plus_1, %y] {in_bounds = [true, true]} : vector<4x4xi32>, memref<32x32xi32, 2>
    
    // CHECK: %[[COMPUTE1:.*]] = arith.addi %[[ITER1]], %[[ITER1]]
    // CHECK: %[[COMPUTE2:.*]] = arith.muli %[[ITER2]], %[[ITER2]]
    // CHECK: scf.yield %[[COMPUTE1]], %[[COMPUTE2]]
  }
  // CHECK: vector.transfer_write %[[LOOP]]#1, %{{.*}}[%[[AFFINE]], %{{.*}}]
  // CHECK: vector.transfer_write %[[LOOP]]#0, %{{.*}}[%{{.*}}, %{{.*}}]
  return
}
