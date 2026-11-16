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

// Test case 3: Hoisting from inner loop where memref is allocated in outer loop
// Tests that we don't incorrectly clone allocations defined outside the hoisted loop
// CHECK-LABEL: @hoist_with_outer_alloc
func.func @hoist_with_outer_alloc() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0_f32 = arith.constant 0.0 : f32
  
  // Outer loop that allocates memory
  // CHECK: scf.for
  scf.for %outer = %c0 to %c4 step %c1 {
    // Allocation happens OUTSIDE the inner loop
    // CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<8x8x8x8xf32, 2>
    %alloc = memref.alloc() : memref<8x8x8x8xf32, 2>
    
    // CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %[[ALLOC]]
    // CHECK-NEXT: %[[INNER_LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[READ]])
    scf.for %inner = %c0 to %c4 step %c1 {
      // Read from the allocation defined in outer loop - should reuse %alloc, not create a new one!
      %val = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_f32 {in_bounds = [true, true, true, true]} : memref<8x8x8x8xf32, 2>, vector<8x8x8x8xf32>
      %result = arith.addf %val, %val : vector<8x8x8x8xf32>
      vector.transfer_write %result, %alloc[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<8x8x8x8xf32>, memref<8x8x8x8xf32, 2>
      // CHECK-NEXT: %[[COMPUTE:.*]] = arith.addf %[[ITER]], %[[ITER]]
      // CHECK-NEXT: scf.yield %[[COMPUTE]]
    }
    // CHECK: vector.transfer_write %[[INNER_LOOP]], %[[ALLOC]]
    
    memref.dealloc %alloc : memref<8x8x8x8xf32, 2>
  }
  return
}

// Test case 4: Two pairs on disjoint rows of the same memref
// CHECK-LABEL: @hoist_two_pairs_from_same_loop
#map1 = affine_map<()[s0] -> (s0 + 4)>
func.func @hoist_two_pairs_from_same_loop(%arg0: memref<32x32xi32, 2>, %x: index, %y: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %x_plus_4 = affine.apply #map1()[%x]

  // CHECK-DAG: %[[READ1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK-DAG: %[[AFFINE2:.*]] = affine.apply
  // CHECK: %[[READ2:.*]] = vector.transfer_read %{{.*}}[%[[AFFINE2]], %{{.*}}]
  // CHECK-NEXT: %[[LOOP:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER1:.*]] = %[[READ1]], %[[ITER2:.*]] = %[[READ2]])
  // expected-remark @below {{hoisted loop}}
  scf.for %i = %c0 to %c4 step %c1 {
    %val1 = vector.transfer_read %arg0[%x, %y], %c0_i32 {in_bounds = [true, true]} : memref<32x32xi32, 2>, vector<4x4xi32>
    %result1 = arith.addi %val1, %val1 : vector<4x4xi32>
    %val2 = vector.transfer_read %arg0[%x_plus_4, %y], %c0_i32 {in_bounds = [true, true]} : memref<32x32xi32, 2>, vector<4x4xi32>
    %result2 = arith.muli %val2, %val2 : vector<4x4xi32>
    vector.transfer_write %result1, %arg0[%x, %y] {in_bounds = [true, true]} : vector<4x4xi32>, memref<32x32xi32, 2>
    vector.transfer_write %result2, %arg0[%x_plus_4, %y] {in_bounds = [true, true]} : vector<4x4xi32>, memref<32x32xi32, 2>
    // CHECK: %[[COMPUTE1:.*]] = arith.addi %[[ITER1]], %[[ITER1]]
    // CHECK: %[[COMPUTE2:.*]] = arith.muli %[[ITER2]], %[[ITER2]]
    // CHECK: scf.yield %[[COMPUTE1]], %[[COMPUTE2]]
  }
  // The writes keep their order in the loop body.
  // CHECK: vector.transfer_write %[[LOOP]]#0, %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK-NEXT: vector.transfer_write %[[LOOP]]#1, %{{.*}}[%[[AFFINE2]], %{{.*}}]
  return
}

// Test case 5: Three pairs whose indices are computed inside the loop. The
// index computations move out ahead of the loop, and the writes keep their
// order in the loop body.
// CHECK-LABEL: @hoist_three_pairs_keeps_write_order
#map2 = affine_map<()[s0] -> (s0 + 8)>
func.func @hoist_three_pairs_keeps_write_order(%arg0: memref<32x32xf32, 2>, %x: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[X1:.*]] = affine.apply #{{.*}}()[%{{.*}}]
  // CHECK: %[[X2:.*]] = affine.apply #{{.*}}()[%{{.*}}]
  // CHECK: %[[LOOP:.*]]:3 = scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    %x1 = affine.apply #map1()[%x]
    %x2 = affine.apply #map2()[%x]
    %v0 = vector.transfer_read %arg0[%x, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    %v1 = vector.transfer_read %arg0[%x1, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    %v2 = vector.transfer_read %arg0[%x2, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    %r0 = arith.addf %v0, %v0 : vector<4x4xf32>
    %r1 = arith.addf %v1, %v1 : vector<4x4xf32>
    %r2 = arith.addf %v2, %v2 : vector<4x4xf32>
    vector.transfer_write %r0, %arg0[%x, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
    vector.transfer_write %r1, %arg0[%x1, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
    vector.transfer_write %r2, %arg0[%x2, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
  }
  // CHECK: vector.transfer_write %[[LOOP]]#0, %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK-NEXT: vector.transfer_write %[[LOOP]]#1, %{{.*}}[%[[X1]], %{{.*}}]
  // CHECK-NEXT: vector.transfer_write %[[LOOP]]#2, %{{.*}}[%[[X2]], %{{.*}}]
  // CHECK-NEXT: return
  return
}

// Test case 6: Pairs on overlapping rows depend on each other across
// iterations, so nothing is hoisted.
// CHECK-LABEL: @keep_overlapping_pairs
#map3 = affine_map<()[s0] -> (s0 + 1)>
func.func @keep_overlapping_pairs(%arg0: memref<32x32xf32, 2>, %x: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %x1 = affine.apply #map3()[%x]
  // CHECK: scf.for
  // CHECK-NOT: iter_args
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_read
  // CHECK: vector.transfer_write
  // CHECK: vector.transfer_write
  // CHECK: }
  scf.for %i = %c0 to %c4 step %c1 {
    %v0 = vector.transfer_read %arg0[%x, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    %v1 = vector.transfer_read %arg0[%x1, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    %r0 = arith.addf %v0, %v0 : vector<4x4xf32>
    %r1 = arith.addf %v1, %v1 : vector<4x4xf32>
    vector.transfer_write %r0, %arg0[%x, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
    vector.transfer_write %r1, %arg0[%x1, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
  }
  return
}

// Test case 7: The read and the write compute the same index separately.
// Other duplicate ops in the body are left alone.
// CHECK-LABEL: @hoist_equivalent_indices
func.func @hoist_equivalent_indices(%arg0: memref<32x32xf32, 2>, %x: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[X:.*]] = affine.apply
  // CHECK: vector.transfer_read %{{.*}}[%[[X]], %{{.*}}]
  // CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args
  scf.for %i = %c0 to %c4 step %c1 {
    %xr = affine.apply #map2()[%x]
    %v = vector.transfer_read %arg0[%xr, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    // CHECK: arith.addf
    // CHECK-NEXT: arith.addf
    %a = arith.addf %v, %v : vector<4x4xf32>
    %b = arith.addf %v, %v : vector<4x4xf32>
    %r = arith.mulf %a, %b : vector<4x4xf32>
    %xw = affine.apply #map2()[%x]
    vector.transfer_write %r, %arg0[%xw, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
  }
  // CHECK: vector.transfer_write %[[LOOP]], %{{.*}}[%[[X]], %{{.*}}]
  return
}

// Test case 8: Only the given loop is rewritten, even when the pair is also
// invariant in the enclosing loop. A read with no matching write stays put.
// CHECK-LABEL: @hoist_only_target_loop
func.func @hoist_only_target_loop(%arg0: memref<32x32xf32, 2>, %arg1: memref<32x32xf32, 2>) -> vector<4x4xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %init = arith.constant dense<0.0> : vector<4x4xf32>
  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<4x4xf32>)
  // CHECK: vector.transfer_read
  // CHECK: scf.for {{.*}} iter_args
  // CHECK: vector.transfer_read %{{.*}}
  // CHECK: scf.yield
  // CHECK: vector.transfer_write
  %res = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (vector<4x4xf32>) {
    // expected-remark @below {{input handle}}
    %in = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %acc) -> (vector<4x4xf32>) {
      %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
      %u = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
      %r = arith.addf %v, %u : vector<4x4xf32>
      %s = arith.addf %a, %u : vector<4x4xf32>
      vector.transfer_write %r, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
      scf.yield %s : vector<4x4xf32>
    }
    scf.yield %in : vector<4x4xf32>
  }
  return %res : vector<4x4xf32>
}

// Test case 9: Pairs that cannot become an iter_arg stay in the loop: a masked
// pair, a pair on a view of another buffer, and a write before its read.
// CHECK-LABEL: @keep_masked_pair
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @keep_masked_pair(%arg0: memref<32x32xf32, 2>, %m: vector<4x4xi1>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %arg0[%c0, %c0], %cst, %m {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    %r = arith.addf %v, %v : vector<4x4xf32>
    vector.transfer_write %r, %arg0[%c0, %c0], %m {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
  }
  return
}

// CHECK-LABEL: @keep_view_base
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @keep_view_base(%arg0: memref<32x32xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %sv = memref.subview %arg0[0, 0] [8, 8] [1, 1] : memref<32x32xf32, 2> to memref<8x8xf32, strided<[32, 1]>, 2>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = vector.transfer_read %sv[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32, strided<[32, 1]>, 2>, vector<4x4xf32>
    %r = arith.addf %v, %v : vector<4x4xf32>
    vector.transfer_write %r, %sv[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<8x8xf32, strided<[32, 1]>, 2>
  }
  return
}

// CHECK-LABEL: @keep_write_before_read
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
func.func @keep_write_before_read(%arg0: memref<32x32xf32, 2>, %arg1: memref<32x32xf32, 2>, %w: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    vector.transfer_write %w, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
    %v = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32, 2>, vector<4x4xf32>
    vector.transfer_write %v, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<32x32xf32, 2>
  }
  return
}
