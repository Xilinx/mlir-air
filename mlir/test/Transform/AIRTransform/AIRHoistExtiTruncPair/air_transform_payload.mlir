//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic hoisting of single extsi/trunci pair
// CHECK-LABEL: @hoist_simple_extsi_trunci
func.func @hoist_simple_extsi_trunci(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[INIT_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK-NEXT: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[INIT_EXT]]) -> (vector<64xi32>)
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    // CHECK-NOT: arith.extsi
    %acc_ext = arith.extsi %acc : vector<64xi16> to vector<64xi32>
    
    // Some computation
    %val = arith.addi %acc_ext, %acc_ext : vector<64xi32>
    
    // CHECK-NOT: arith.trunci
    %val_trunc = arith.trunci %val : vector<64xi32> to vector<64xi16>
    scf.yield %val_trunc : vector<64xi16>
    // CHECK: scf.yield %{{.*}} : vector<64xi32>
  }
  // CHECK: %[[RESULT:.*]] = arith.trunci %[[LOOP]] : vector<64xi32> to vector<64xi16>
  // CHECK: return %[[RESULT]]
  return %result : vector<64xi16>
}

// Test case 2: Hoisting with vector.shape_cast operations
// CHECK-LABEL: @hoist_with_shape_cast
func.func @hoist_with_shape_cast(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[INIT_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK-NEXT: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[INIT_EXT]]) -> (vector<64xi32>)
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    %acc_shaped = vector.shape_cast %acc : vector<64xi16> to vector<1x1x8x8xi16>
    // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[ITER]] : vector<64xi32> to vector<1x1x8x8xi32>
    // CHECK-NOT: arith.extsi
    %acc_ext = arith.extsi %acc_shaped : vector<1x1x8x8xi16> to vector<1x1x8x8xi32>
    
    // Computation
    %val = arith.addi %acc_ext, %acc_ext : vector<1x1x8x8xi32>
    
    // CHECK-NOT: arith.trunci
    %val_trunc = arith.trunci %val : vector<1x1x8x8xi32> to vector<1x1x8x8xi16>
    %val_flat = vector.shape_cast %val_trunc : vector<1x1x8x8xi16> to vector<64xi16>
    scf.yield %val_flat : vector<64xi16>
    // CHECK: %[[FLAT:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi32> to vector<64xi32>
    // CHECK: scf.yield %[[FLAT]] : vector<64xi32>
  }
  // CHECK: %[[RESULT:.*]] = arith.trunci %[[LOOP]] : vector<64xi32> to vector<64xi16>
  // CHECK: return %[[RESULT]]
  return %result : vector<64xi16>
}

// Test case 3: Hoisting four pairs from the same loop (similar to the original example)
// CHECK-LABEL: @hoist_four_pairs
func.func @hoist_four_pairs(%arg0: vector<64xi16>, %arg1: vector<64xi16>, %arg2: vector<64xi16>, %arg3: vector<64xi16>) -> (vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[INIT0_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK: %[[INIT1_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK: %[[INIT2_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK: %[[INIT3_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK: %[[LOOP:.*]]:4 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER0:.*]] = %[[INIT0_EXT]], %[[ITER1:.*]] = %[[INIT1_EXT]], %[[ITER2:.*]] = %[[INIT2_EXT]], %[[ITER3:.*]] = %[[INIT3_EXT]]) -> (vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)
  %result:4 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc0 = %arg0, %acc1 = %arg1, %acc2 = %arg2, %acc3 = %arg3) -> (vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>) {
    %acc0_shaped = vector.shape_cast %acc0 : vector<64xi16> to vector<1x1x8x8xi16>
    %acc1_shaped = vector.shape_cast %acc1 : vector<64xi16> to vector<1x1x8x8xi16>
    %acc2_shaped = vector.shape_cast %acc2 : vector<64xi16> to vector<1x1x8x8xi16>
    %acc3_shaped = vector.shape_cast %acc3 : vector<64xi16> to vector<1x1x8x8xi16>
    
    // CHECK-DAG: %[[SHAPED0:.*]] = vector.shape_cast %[[ITER0]] : vector<64xi32> to vector<1x1x8x8xi32>
    // CHECK-DAG: %[[SHAPED1:.*]] = vector.shape_cast %[[ITER1]] : vector<64xi32> to vector<1x1x8x8xi32>
    // CHECK-DAG: %[[SHAPED2:.*]] = vector.shape_cast %[[ITER2]] : vector<64xi32> to vector<1x1x8x8xi32>
    // CHECK-DAG: %[[SHAPED3:.*]] = vector.shape_cast %[[ITER3]] : vector<64xi32> to vector<1x1x8x8xi32>
    
    // CHECK-NOT: arith.extsi
    %acc0_ext = arith.extsi %acc0_shaped : vector<1x1x8x8xi16> to vector<1x1x8x8xi32>
    %acc1_ext = arith.extsi %acc1_shaped : vector<1x1x8x8xi16> to vector<1x1x8x8xi32>
    %acc2_ext = arith.extsi %acc2_shaped : vector<1x1x8x8xi16> to vector<1x1x8x8xi32>
    %acc3_ext = arith.extsi %acc3_shaped : vector<1x1x8x8xi16> to vector<1x1x8x8xi32>
    
    // Computation
    %val0 = arith.addi %acc0_ext, %acc0_ext : vector<1x1x8x8xi32>
    %val1 = arith.addi %acc1_ext, %acc1_ext : vector<1x1x8x8xi32>
    %val2 = arith.addi %acc2_ext, %acc2_ext : vector<1x1x8x8xi32>
    %val3 = arith.addi %acc3_ext, %acc3_ext : vector<1x1x8x8xi32>
    
    // CHECK-NOT: arith.trunci
    %val0_trunc = arith.trunci %val0 : vector<1x1x8x8xi32> to vector<1x1x8x8xi16>
    %val1_trunc = arith.trunci %val1 : vector<1x1x8x8xi32> to vector<1x1x8x8xi16>
    %val2_trunc = arith.trunci %val2 : vector<1x1x8x8xi32> to vector<1x1x8x8xi16>
    %val3_trunc = arith.trunci %val3 : vector<1x1x8x8xi32> to vector<1x1x8x8xi16>
    
    %val0_flat = vector.shape_cast %val0_trunc : vector<1x1x8x8xi16> to vector<64xi16>
    %val1_flat = vector.shape_cast %val1_trunc : vector<1x1x8x8xi16> to vector<64xi16>
    %val2_flat = vector.shape_cast %val2_trunc : vector<1x1x8x8xi16> to vector<64xi16>
    %val3_flat = vector.shape_cast %val3_trunc : vector<1x1x8x8xi16> to vector<64xi16>
    
    scf.yield %val0_flat, %val1_flat, %val2_flat, %val3_flat : vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>
    // CHECK: %[[FLAT0:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi32> to vector<64xi32>
    // CHECK: %[[FLAT1:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi32> to vector<64xi32>
    // CHECK: %[[FLAT2:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi32> to vector<64xi32>
    // CHECK: %[[FLAT3:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi32> to vector<64xi32>
    // CHECK: scf.yield %[[FLAT0]], %[[FLAT1]], %[[FLAT2]], %[[FLAT3]] : vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>
  }
  // CHECK-DAG: %{{.*}} = arith.trunci %[[LOOP]]#0 : vector<64xi32> to vector<64xi16>
  // CHECK-DAG: %{{.*}} = arith.trunci %[[LOOP]]#1 : vector<64xi32> to vector<64xi16>
  // CHECK-DAG: %{{.*}} = arith.trunci %[[LOOP]]#2 : vector<64xi32> to vector<64xi16>
  // CHECK-DAG: %{{.*}} = arith.trunci %[[LOOP]]#3 : vector<64xi32> to vector<64xi16>
  // CHECK: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>
  return %result#0, %result#1, %result#2, %result#3 : vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>
}
