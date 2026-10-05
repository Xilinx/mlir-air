//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic hoisting with 2D memref - loop-invariant indices. The
// 4x16 block spans whole rows, so it is contiguous and flattens exactly.
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
    // CHECK-NEXT: %[[SHAPED:.*]] = vector.shape_cast %[[FLAT_READ]] : vector<64xf32> to vector<4x16xf32>
    %val = vector.transfer_read %arg0[%c2, %c0], %c0_f32 {in_bounds = [true, true]} : memref<16x16xf32, 2>, vector<4x16xf32>
    
    %result = arith.addf %val, %val : vector<4x16xf32>
    
    // CHECK: %[[PTR2:.*]] = affine.apply
    // CHECK: %[[FLAT_VAL:.*]] = vector.shape_cast %{{.*}} : vector<4x16xf32> to vector<64xf32>
    // CHECK: vector.transfer_write %[[FLAT_VAL]], %[[COLLAPSED2]][%[[PTR2]]]
    vector.transfer_write %result, %arg0[%c2, %c0] {in_bounds = [true, true]} : vector<4x16xf32>, memref<16x16xf32, 2>
  }
  return
}

// Test case 2: Hoisting with loop IV-dependent indices. The 8x8 block spans
// whole rows of the 32x8 buffer, so it is contiguous and walks one row (8
// elements) per iteration.
// CHECK-LABEL: @hoist_with_iv_dependent_indices
func.func @hoist_with_iv_dependent_indices(%arg0: memref<32x8xi16, 2>) {
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
    // CHECK: %[[STRIDE:.*]] = arith.constant 8 : index
    // CHECK: %[[NEXT_PTR:.*]] = arith.addi %[[PTR]], %[[STRIDE]]
    %val = vector.transfer_read %arg0[%i, %c0], %c0_i16 {in_bounds = [true, true]} : memref<32x8xi16, 2>, vector<8x8xi16>
    
    %result = arith.addi %val, %val : vector<8x8xi16>
    
    // CHECK: %[[FLAT_VAL:.*]] = vector.shape_cast %{{.*}} : vector<8x8xi16> to vector<64xi16>
    // CHECK: vector.transfer_write %[[FLAT_VAL]], %[[COLLAPSED2]][%[[PTR2]]]
    // CHECK: %[[STRIDE2:.*]] = arith.constant 8 : index
    // CHECK: %[[NEXT_PTR2:.*]] = arith.addi %[[PTR2]], %[[STRIDE2]]
    // CHECK: scf.yield %[[NEXT_PTR]], %[[NEXT_PTR2]]
    vector.transfer_write %result, %arg0[%i, %c0] {in_bounds = [true, true]} : vector<8x8xi16>, memref<32x8xi16, 2>
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

// Test case: a non-unit loop step. The pointer advances by step * the row
// size, not by the row size alone.
// CHECK-LABEL: @hoist_non_unit_step
func.func @hoist_non_unit_step(%arg0: memref<8x8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0_i16 = arith.constant 0 : i16
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR:.*]] = %{{.*}}, %[[PTR2:.*]] = %{{.*}}) -> (index, index)
  scf.for %i = %c0 to %c8 step %c4 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR]]]{{.*}} : memref<64xi16, 2>, vector<32xi16>
    // CHECK: %[[STRIDE:.*]] = arith.constant 32 : index
    // CHECK: arith.addi %[[PTR]], %[[STRIDE]]
    %val = vector.transfer_read %arg0[%i, %c0], %c0_i16 {in_bounds = [true]} : memref<8x8xi16, 2>, vector<32xi16>
    %result = arith.addi %val, %val : vector<32xi16>
    vector.transfer_write %result, %arg0[%i, %c0] {in_bounds = [true]} : vector<32xi16>, memref<8x8xi16, 2>
  }
  return
}

// Test case: a broadcasting transfer cannot be walked by a flat pointer, so
// the loop is left as it is.
// CHECK-LABEL: @no_hoist_broadcast_read
func.func @no_hoist_broadcast_read(%arg0: memref<8x1x8xf32, 2>, %arg1: memref<8x4x8xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.0 : f32
  // CHECK-NOT: memref.collapse_shape
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK: vector.transfer_read %arg0[%[[I]], %{{.*}}, %{{.*}}]{{.*}}permutation_map
    %s = vector.transfer_read %arg0[%i, %c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, 0, d2)>} : memref<8x1x8xf32, 2>, vector<1x4x8xf32>
    vector.transfer_write %s, %arg1[%i, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x8xf32>, memref<8x4x8xf32, 2>
  }
  return
}

// Test case: a 4x8 block of an 8x64 buffer is not contiguous, so it cannot be
// read as 32 consecutive elements; the loop is left as it is.
// CHECK-LABEL: @no_hoist_non_contiguous_block
func.func @no_hoist_non_contiguous_block(%arg0: memref<8x64xi16, 2>, %arg1: memref<8x8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0_i16 = arith.constant 0 : i16
  // CHECK-NOT: memref.collapse_shape
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  scf.for %i = %c0 to %c8 step %c4 {
    // CHECK: vector.transfer_read %arg0[%[[I]], %{{.*}}]{{.*}} : memref<8x64xi16, 2>, vector<4x8xi16>
    %v = vector.transfer_read %arg0[%i, %c0], %c0_i16 {in_bounds = [true, true]} : memref<8x64xi16, 2>, vector<4x8xi16>
    vector.transfer_write %v, %arg1[%i, %c0] {in_bounds = [true, true]} : vector<4x8xi16>, memref<8x8xi16, 2>
  }
  return
}

// Test case: an index `iv + 1` starts the walk at row 1, not at row 0.
// CHECK-LABEL: @hoist_iv_plus_constant
func.func @hoist_iv_plus_constant(%arg0: memref<16x8xi16, 2>, %arg1: memref<16x8xi16, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i16 = arith.constant 0 : i16
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[ROW:.*]] = arith.addi %{{.*}}, %[[C1]] : index
  // CHECK: %[[BASE:.*]] = affine.apply #{{.*}}(%[[ROW]], %{{.*}})
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[BASE]],
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.addi %i, %c1 : index
    %v = vector.transfer_read %arg0[%r, %c0], %c0_i16 {in_bounds = [true]} : memref<16x8xi16, 2>, vector<8xi16>
    vector.transfer_write %v, %arg1[%i, %c0] {in_bounds = [true]} : vector<8xi16>, memref<16x8xi16, 2>
  }
  return
}

// Test case: a loop with no IV-dependent transfer. Its invariant transfers
// are flattened only where that is exact: the broadcasting read keeps its
// permutation map, the plain write is flattened.
// CHECK-LABEL: @no_flatten_invariant_broadcast
func.func @no_flatten_invariant_broadcast(%arg0: memref<1x8xf32, 2>, %arg1: memref<4x8xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK: vector.transfer_read %arg0[%{{.*}}, %{{.*}}]{{.*}}permutation_map
    %s = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (0, d1)>} : memref<1x8xf32, 2>, vector<4x8xf32>
    // CHECK: vector.transfer_write %{{.*}} : vector<32xf32>, memref<32xf32, 2>
    vector.transfer_write %s, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32, 2>
  }
  return
}

// Test case: a loop with no IV-dependent transfer, whose invariant 4x4 block
// at column 2 of a 16x16 buffer is not contiguous: it is not flattened into 16
// consecutive elements.
// CHECK-LABEL: @no_flatten_invariant_block
// CHECK-NOT: memref.collapse_shape %arg0
// CHECK: vector.transfer_read %arg0[%{{.*}}, %{{.*}}]{{.*}} : memref<16x16xf32, 2>, vector<4x4xf32>
func.func @no_flatten_invariant_block(%arg0: memref<16x16xf32, 2>, %arg1: memref<4x4xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %arg0[%c0, %c2], %cst {in_bounds = [true, true]} : memref<16x16xf32, 2>, vector<4x4xf32>
    vector.transfer_write %v, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32, 2>
  }
  return
}
