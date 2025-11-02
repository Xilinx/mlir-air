//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic redundant transfer_read elimination
// CHECK-LABEL: @eliminate_simple_redundant_read
func.func @eliminate_simple_redundant_read(%memref: memref<8x8xi32, 2>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<1> : vector<4xi32>
  %other = arith.constant dense<2> : vector<4xi32>
  
  // CHECK: %[[READ:.*]] = vector.transfer_read %{{.*}}[%c0, %c2]
  // First read
  %0 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  // CHECK: %{{.*}} = arith.addi %[[READ]], %{{.*}}
  %1 = arith.addi %0, %cst : vector<4xi32>
  
  // Redundant read - should be eliminated
  // CHECK-NOT: vector.transfer_read
  %2 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  // CHECK: %{{.*}} = arith.muli %[[READ]], %{{.*}}
  %3 = arith.muli %2, %other : vector<4xi32>
  // CHECK: return
  
  return %1, %3 : vector<4xi32>, vector<4xi32>
}

// Test case 2: Reads with write in between - should NOT be eliminated
// CHECK-LABEL: @keep_read_after_write
func.func @keep_read_after_write(%memref: memref<8x8xi32, 2>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<1> : vector<4xi32>
  
  // CHECK: %[[READ1:.*]] = vector.transfer_read %{{.*}}[%c0, %c2]
  // First read
  %0 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  %1 = arith.addi %0, %cst : vector<4xi32>
  
  // Write to the same memref
  // CHECK: vector.transfer_write
  vector.transfer_write %1, %memref[%c0, %c2] {in_bounds = [true]} : vector<4xi32>, memref<8x8xi32, 2>
  
  // Second read - should NOT be eliminated because of write in between
  // CHECK: %[[READ2:.*]] = vector.transfer_read %{{.*}}[%c0, %c2]
  %2 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  // CHECK: arith.muli %[[READ2]]
  %3 = arith.muli %2, %cst : vector<4xi32>
  
  return %1, %3 : vector<4xi32>, vector<4xi32>
}

// Test case 3: Reads with different indices - should NOT be eliminated
// CHECK-LABEL: @keep_different_indices
func.func @keep_different_indices(%memref: memref<8x8xi32, 2>) -> (vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  
  // First read at [0, 2]
  %0 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  
  // Second read at [1, 2] - different index, should NOT be eliminated
  // CHECK: %{{.*}} = vector.transfer_read %{{.*}}[%c0, %c2]
  // CHECK: %{{.*}} = vector.transfer_read %{{.*}}[%c1, %c2]
  %1 = vector.transfer_read %memref[%c1, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// Test case 4: Reads with different result types - should NOT be eliminated
// CHECK-LABEL: @keep_different_types
func.func @keep_different_types(%memref: memref<8x8xi32, 2>) -> (vector<4xi32>, vector<8xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  
  // First read - vector<4xi32>
  %0 = vector.transfer_read %memref[%c0, %c0], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  
  // Second read - vector<8xi32>, different type, should NOT be eliminated
  // CHECK: %{{.*}} = vector.transfer_read %{{.*}}[%c0, %c0]{{.*}}: memref<8x8xi32, 2>, vector<4xi32>
  // CHECK: %{{.*}} = vector.transfer_read %{{.*}}[%c0, %c0]{{.*}}: memref<8x8xi32, 2>, vector<8xi32>
  %1 = vector.transfer_read %memref[%c0, %c0], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<8xi32>
  
  return %0, %1 : vector<4xi32>, vector<8xi32>
}

// Test case 5: Multiple redundant reads
// CHECK-LABEL: @eliminate_multiple_redundant_reads
func.func @eliminate_multiple_redundant_reads(%memref: memref<8x8xi32, 2>) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<1> : vector<4xi32>
  
  // CHECK: %[[READ:.*]] = vector.transfer_read
  // First read
  %0 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  // CHECK-NEXT: arith.addi %[[READ]]
  %1 = arith.addi %0, %cst : vector<4xi32>
  
  // Redundant read 1 - should be eliminated
  // CHECK-NOT: vector.transfer_read
  %2 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  // CHECK: arith.muli %[[READ]]
  %3 = arith.muli %2, %cst : vector<4xi32>
  
  // Redundant read 2 - should be eliminated
  // CHECK-NOT: vector.transfer_read
  %4 = vector.transfer_read %memref[%c0, %c2], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  // CHECK: arith.addi
  %5 = arith.addi %4, %3 : vector<4xi32>
  
  return %1, %3, %5 : vector<4xi32>, vector<4xi32>, vector<4xi32>
}

// Test case 6: Redundant reads in a loop (simulating unrolled loop)
// CHECK-LABEL: @eliminate_in_loop_pattern
func.func @eliminate_in_loop_pattern(%memref: memref<16x16xf32, 2>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.0 : f32
  %init = arith.constant dense<0.0> : vector<4xf32>
  
  // Simulating an unrolled loop where the same read appears multiple times
  %0 = vector.transfer_read %memref[%c0, %c0], %cst {in_bounds = [true]} : memref<16x16xf32, 2>, vector<4xf32>
  %1 = vector.transfer_read %memref[%c4, %c0], %cst {in_bounds = [true]} : memref<16x16xf32, 2>, vector<4xf32>
  %acc1 = arith.addf %0, %1 : vector<4xf32>
  
  // Redundant reads
  %2 = vector.transfer_read %memref[%c0, %c0], %cst {in_bounds = [true]} : memref<16x16xf32, 2>, vector<4xf32>
  %3 = vector.transfer_read %memref[%c4, %c0], %cst {in_bounds = [true]} : memref<16x16xf32, 2>, vector<4xf32>
  %acc2 = arith.addf %2, %3 : vector<4xf32>
  
  %result = arith.addf %acc1, %acc2 : vector<4xf32>
  
  // CHECK: %[[R0:.*]] = vector.transfer_read %{{.*}}[%c0, %c0]
  // CHECK-NEXT: %[[R1:.*]] = vector.transfer_read %{{.*}}[%c4, %c0]
  // CHECK-NEXT: %{{.*}} = arith.addf %[[R0]], %[[R1]]
  // CHECK-NOT: vector.transfer_read
  // CHECK: %{{.*}} = arith.addf %[[R0]], %[[R1]]
  // CHECK: %{{.*}} = arith.addf
  
  return %result : vector<4xf32>
}

// Test case 7: Redundant reads with affine indices
// CHECK-LABEL: @eliminate_with_affine_indices
#map = affine_map<()[s0] -> (s0 * 2)>
func.func @eliminate_with_affine_indices(%memref: memref<32x32xi16, 2>, %x: index) -> (vector<8xi16>, vector<8xi16>) {
  %c0 = arith.constant 0 : index
  %c0_i16 = arith.constant 0 : i16
  %idx = affine.apply #map()[%x]
  
  // First read
  %0 = vector.transfer_read %memref[%idx, %c0], %c0_i16 {in_bounds = [true]} : memref<32x32xi16, 2>, vector<8xi16>
  
  // Redundant read with same affine index - should be eliminated
  %1 = vector.transfer_read %memref[%idx, %c0], %c0_i16 {in_bounds = [true]} : memref<32x32xi16, 2>, vector<8xi16>
  
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %{{.*}}[%[[IDX]], %c0]
  // CHECK-NOT: vector.transfer_read
  // CHECK: return
  
  return %0, %1 : vector<8xi16>, vector<8xi16>
}

// Test case 8: Mixed case - some redundant, some not
// CHECK-LABEL: @mixed_redundant_and_unique
func.func @mixed_redundant_and_unique(%memref: memref<16x16xi32, 2>) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  
  // Read at [0, 0]
  %0 = vector.transfer_read %memref[%c0, %c0], %c0_i32 {in_bounds = [true]} : memref<16x16xi32, 2>, vector<4xi32>
  
  // Read at [4, 0] - different location
  %1 = vector.transfer_read %memref[%c4, %c0], %c0_i32 {in_bounds = [true]} : memref<16x16xi32, 2>, vector<4xi32>
  
  // Redundant read at [0, 0] - should be eliminated
  %2 = vector.transfer_read %memref[%c0, %c0], %c0_i32 {in_bounds = [true]} : memref<16x16xi32, 2>, vector<4xi32>
  
  // Redundant read at [4, 0] - should be eliminated
  %3 = vector.transfer_read %memref[%c4, %c0], %c0_i32 {in_bounds = [true]} : memref<16x16xi32, 2>, vector<4xi32>
  
  // CHECK: %[[R0:.*]] = vector.transfer_read %{{.*}}[%c0, %c0]
  // CHECK-NEXT: %[[R1:.*]] = vector.transfer_read %{{.*}}[%c4, %c0]
  // CHECK-NOT: vector.transfer_read
  // CHECK: return
  
  return %0, %1, %2, %3 : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
}

// Test case 9: Redundant reads with 2D vectors
// CHECK-LABEL: @eliminate_2d_vector_reads
func.func @eliminate_2d_vector_reads(%memref: memref<16x16xf32, 2>) -> (vector<4x4xf32>, vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  
  // First 2D read
  %0 = vector.transfer_read %memref[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf32, 2>, vector<4x4xf32>
  
  // Redundant 2D read - should be eliminated
  %1 = vector.transfer_read %memref[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf32, 2>, vector<4x4xf32>
  
  // CHECK: %[[READ:.*]] = vector.transfer_read %{{.*}}[%c0, %c0]{{.*}}: memref<16x16xf32, 2>, vector<4x4xf32>
  // CHECK-NOT: vector.transfer_read
  // CHECK: return
  
  return %0, %1 : vector<4x4xf32>, vector<4x4xf32>
}

// Test case 10: No redundancy - all reads should remain
// CHECK-LABEL: @keep_all_unique_reads
func.func @keep_all_unique_reads(%memref: memref<8x8xi32, 2>) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  
  // All reads from different locations - none should be eliminated
  %0 = vector.transfer_read %memref[%c0, %c0], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  %1 = vector.transfer_read %memref[%c1, %c0], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  %2 = vector.transfer_read %memref[%c2, %c0], %c0_i32 {in_bounds = [true]} : memref<8x8xi32, 2>, vector<4xi32>
  
  // CHECK: vector.transfer_read %{{.*}}[%c0, %c0]
  // CHECK: vector.transfer_read %{{.*}}[%c1, %c0]
  // CHECK: vector.transfer_read %{{.*}}[%c2, %c0]
  
  return %0, %1, %2 : vector<4xi32>, vector<4xi32>, vector<4xi32>
}
