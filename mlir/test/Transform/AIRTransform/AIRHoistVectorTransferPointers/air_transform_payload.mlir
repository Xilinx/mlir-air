//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic hoisting with 2D memref - loop-invariant indices
// CHECK-LABEL: @hoist_simple_2d_transfers
func.func @hoist_simple_2d_transfers(%arg0: memref<16x16xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_f32 = arith.constant 0.0 : f32
  
  // CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %{{.*}}
  // CHECK: %[[COLLAPSED2:.*]] = memref.collapse_shape %{{.*}}
  // CHECK: scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: %[[PTR:.*]] = affine.apply
    // CHECK: %[[FLAT_READ:.*]] = vector.transfer_read %[[COLLAPSED]][%[[PTR]]]
    // CHECK-NEXT: %[[SHAPED:.*]] = vector.shape_cast %[[FLAT_READ]] : vector<16xf32> to vector<4x4xf32>
    %val = vector.transfer_read %arg0[%c0, %c2], %c0_f32 {in_bounds = [true, true]} : memref<16x16xf32, 2>, vector<4x4xf32>
    
    %result = arith.addf %val, %val : vector<4x4xf32>
    
    // CHECK: %[[PTR2:.*]] = affine.apply
    // CHECK: %[[FLAT_VAL:.*]] = vector.shape_cast %{{.*}} : vector<4x4xf32> to vector<16xf32>
    // CHECK: vector.transfer_write %[[FLAT_VAL]], %[[COLLAPSED2]][%[[PTR2]]]
    vector.transfer_write %result, %arg0[%c0, %c2] {in_bounds = [true, true]} : vector<4x4xf32>, memref<16x16xf32, 2>
  }
  return
}

// Test case 2: Hoisting with loop IV-dependent indices
// CHECK-LABEL: @hoist_with_iv_dependent_indices
func.func @hoist_with_iv_dependent_indices(%arg0: memref<32x32xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i16 = arith.constant 0 : i16
  
  // CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %{{.*}}
  // CHECK: %[[BASE_PTR:.*]] = affine.apply
  // CHECK: %[[COLLAPSED2:.*]] = memref.collapse_shape %{{.*}}
  // CHECK: %[[BASE_PTR2:.*]] = affine.apply
  // CHECK: scf.for %[[IV:.*]] = {{.*}} iter_args(%[[PTR:.*]] = %[[BASE_PTR]], %[[PTR2:.*]] = %[[BASE_PTR2]]) -> (index, index)
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK: %[[FLAT_READ:.*]] = vector.transfer_read %[[COLLAPSED]][%[[PTR]]]
    // CHECK-NEXT: %[[SHAPED:.*]] = vector.shape_cast %[[FLAT_READ]] : vector<64xi16> to vector<8x8xi16>
    // CHECK: %[[STRIDE:.*]] = arith.constant 1 : index
    // CHECK: %[[NEXT_PTR:.*]] = arith.addi %[[PTR]], %[[STRIDE]]
    %val = vector.transfer_read %arg0[%c0, %i], %c0_i16 {in_bounds = [true, true]} : memref<32x32xi16, 2>, vector<8x8xi16>
    
    %result = arith.addi %val, %val : vector<8x8xi16>
    
    // CHECK: %[[FLAT_VAL:.*]] = vector.shape_cast %{{.*}} : vector<8x8xi16> to vector<64xi16>
    // CHECK: vector.transfer_write %[[FLAT_VAL]], %[[COLLAPSED2]][%[[PTR2]]]
    // CHECK: %[[STRIDE2:.*]] = arith.constant 1 : index
    // CHECK: %[[NEXT_PTR2:.*]] = arith.addi %[[PTR2]], %[[STRIDE2]]
    // CHECK: scf.yield %[[NEXT_PTR]], %[[NEXT_PTR2]]
    vector.transfer_write %result, %arg0[%c0, %i] {in_bounds = [true, true]} : vector<8x8xi16>, memref<32x32xi16, 2>
  }
  return
}

// Test case 3: IV in higher dimension (tests correct stride calculation)
// CHECK-LABEL: @hoist_iv_in_higher_dimension
func.func @hoist_iv_in_higher_dimension(%arg0: memref<8x8x8x8xi8, 2>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i8 = arith.constant 0 : i8
  
  // CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %{{.*}}
  // CHECK: %[[BASE_PTR:.*]] = affine.apply
  // CHECK: %[[COLLAPSED2:.*]] = memref.collapse_shape %{{.*}}
  // CHECK: %[[BASE_PTR2:.*]] = affine.apply
  // CHECK: scf.for %[[IV:.*]] = {{.*}} iter_args(%[[PTR:.*]] = %[[BASE_PTR]], %[[PTR2:.*]] = %[[BASE_PTR2]]) -> (index, index)
  scf.for %i = %c0 to %c8 step %c1 {
    // For memref<8x8x8x8xi8>, IV in dim0 should use stride=512 (8*8*8)
    // CHECK: %[[FLAT_READ:.*]] = vector.transfer_read %[[COLLAPSED]][%[[PTR]]]
    // CHECK-NEXT: %[[SHAPED:.*]] = vector.shape_cast %[[FLAT_READ]] : vector<64xi8> to vector<1x1x8x8xi8>
    // CHECK: %[[STRIDE:.*]] = arith.constant 512 : index
    // CHECK: %[[NEXT_PTR:.*]] = arith.addi %[[PTR]], %[[STRIDE]]
    %val = vector.transfer_read %arg0[%i, %arg1, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<8x8x8x8xi8, 2>, vector<1x1x8x8xi8>
    
    %result = arith.addi %val, %val : vector<1x1x8x8xi8>
    
    // CHECK: %[[FLAT_VAL:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi8> to vector<64xi8>
    // CHECK: vector.transfer_write %[[FLAT_VAL]], %[[COLLAPSED2]][%[[PTR2]]]
    // CHECK: %[[STRIDE2:.*]] = arith.constant 512 : index
    // CHECK: %[[NEXT_PTR2:.*]] = arith.addi %[[PTR2]], %[[STRIDE2]]
    // CHECK: scf.yield %[[NEXT_PTR]], %[[NEXT_PTR2]]
    vector.transfer_write %result, %arg0[%i, %arg1, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xi8>, memref<8x8x8x8xi8, 2>
  }
  return
}
