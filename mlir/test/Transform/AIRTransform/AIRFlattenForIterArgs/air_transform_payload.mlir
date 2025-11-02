//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic flattening with single vector iter_arg
// CHECK-LABEL: @flatten_single_vector_iter_arg
func.func @flatten_single_vector_iter_arg(%v0: vector<1x1x8x8xi16>) -> vector<1x1x8x8xi16> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[FLAT:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
  // CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[FLAT]]) -> (vector<64xi16>)
  %result = scf.for %i = %c0 to %c4 step %c1 
      iter_args(%arg0 = %v0) 
      -> (vector<1x1x8x8xi16>) {
    // CHECK: %[[SHAPED:.*]] = vector.shape_cast %[[ARG]] : vector<64xi16> to vector<1x1x8x8xi16>
    // Some computation using the shaped value
    %temp = arith.addi %arg0, %arg0 : vector<1x1x8x8xi16>
    // CHECK: %[[TEMP:.*]] = arith.addi %[[SHAPED]], %[[SHAPED]]
    // CHECK: %[[TEMP_FLAT:.*]] = vector.shape_cast %[[TEMP]] : vector<1x1x8x8xi16> to vector<64xi16>
    // CHECK: scf.yield %[[TEMP_FLAT]] : vector<64xi16>
    scf.yield %temp : vector<1x1x8x8xi16>
  }
  
  return %result : vector<1x1x8x8xi16>
}

// Test case 2: Flattening with multiple vector iter_args (as in the example from the TD file)
// CHECK-LABEL: @flatten_multiple_vector_iter_args
func.func @flatten_multiple_vector_iter_args(
    %v0: vector<1x1x8x8xi16>, %v1: vector<1x1x8x8xi16>, 
    %v2: vector<1x1x8x8xi16>, %v3: vector<1x1x8x8xi16>) 
    -> (vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK-DAG: %[[V0_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
  // CHECK-DAG: %[[V1_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
  // CHECK-DAG: %[[V2_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
  // CHECK-DAG: %[[V3_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
  // CHECK: %{{.*}}:4 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[V0_FLAT]], %{{.*}} = %[[V1_FLAT]], %{{.*}} = %[[V2_FLAT]], %{{.*}} = %[[V3_FLAT]]) -> (vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>)
  %result:4 = scf.for %i = %c0 to %c4 step %c1 
      iter_args(%arg0 = %v0, %arg1 = %v1, %arg2 = %v2, %arg3 = %v3) 
      -> (vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>) {
    // CHECK-DAG: %[[ARG0_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<64xi16> to vector<1x1x8x8xi16>
    // CHECK-DAG: %[[ARG1_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<64xi16> to vector<1x1x8x8xi16>
    // CHECK-DAG: %[[ARG2_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<64xi16> to vector<1x1x8x8xi16>
    // CHECK-DAG: %[[ARG3_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<64xi16> to vector<1x1x8x8xi16>
    
    // Some computation
    %r0 = arith.addi %arg0, %arg1 : vector<1x1x8x8xi16>
    %r1 = arith.addi %arg1, %arg2 : vector<1x1x8x8xi16>
    %r2 = arith.addi %arg2, %arg3 : vector<1x1x8x8xi16>
    %r3 = arith.addi %arg3, %arg0 : vector<1x1x8x8xi16>
    
    // CHECK-DAG: %{{.*}} = arith.addi %[[ARG0_SHAPED]], %[[ARG1_SHAPED]]
    // CHECK-DAG: %{{.*}} = arith.addi %[[ARG1_SHAPED]], %[[ARG2_SHAPED]]
    // CHECK-DAG: %{{.*}} = arith.addi %[[ARG2_SHAPED]], %[[ARG3_SHAPED]]
    // CHECK-DAG: %{{.*}} = arith.addi %[[ARG3_SHAPED]], %[[ARG0_SHAPED]]
    // CHECK-DAG: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
    // CHECK-DAG: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
    // CHECK-DAG: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
    // CHECK-DAG: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x1x8x8xi16> to vector<64xi16>
    // CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi16>, vector<64xi16>, vector<64xi16>, vector<64xi16>
    scf.yield %r0, %r1, %r2, %r3 : vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>
  }
  
  return %result#0, %result#1, %result#2, %result#3 : vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>
}

// Test case 3: Mixed scalar and vector iter_args (only vectors should be flattened)
// CHECK-LABEL: @flatten_mixed_iter_args
func.func @flatten_mixed_iter_args(%v0: vector<2x4xi32>, %s0: i32) -> (vector<2x4xi32>, i32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[V0_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<2x4xi32> to vector<8xi32>
  // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[V0_FLAT]], %{{.*}} = %{{.*}}) -> (vector<8xi32>, i32)
  %result:2 = scf.for %i = %c0 to %c4 step %c1 
      iter_args(%arg0 = %v0, %arg1 = %s0) 
      -> (vector<2x4xi32>, i32) {
    // CHECK: %[[ARG0_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<8xi32> to vector<2x4xi32>
    
    %temp_v = arith.addi %arg0, %arg0 : vector<2x4xi32>
    %temp_s = arith.addi %arg1, %arg1 : i32
    
    // CHECK: %{{.*}} = arith.addi %[[ARG0_SHAPED]], %[[ARG0_SHAPED]]
    // CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<2x4xi32> to vector<8xi32>
    // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<8xi32>, i32
    scf.yield %temp_v, %temp_s : vector<2x4xi32>, i32
  }
  
  return %result#0, %result#1 : vector<2x4xi32>, i32
}

// Test case 4: Different vector shapes and types
// CHECK-LABEL: @flatten_different_vector_types
func.func @flatten_different_vector_types(%v0: vector<4x4xf32>, %v1: vector<2x2x2xf16>) 
    -> (vector<4x4xf32>, vector<2x2x2xf16>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  
  // CHECK-DAG: %[[V0_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<4x4xf32> to vector<16xf32>
  // CHECK-DAG: %[[V1_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<2x2x2xf16> to vector<8xf16>
  // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[V0_FLAT]], %{{.*}} = %[[V1_FLAT]]) -> (vector<16xf32>, vector<8xf16>)
  %result:2 = scf.for %i = %c0 to %c2 step %c1 
      iter_args(%arg0 = %v0, %arg1 = %v1) 
      -> (vector<4x4xf32>, vector<2x2x2xf16>) {
    // CHECK: %[[ARG0_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<16xf32> to vector<4x4xf32>
    // CHECK: %[[ARG1_SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<8xf16> to vector<2x2x2xf16>
    
    %r0 = arith.addf %arg0, %arg0 : vector<4x4xf32>
    %r1 = arith.addf %arg1, %arg1 : vector<2x2x2xf16>
    
    // CHECK: %{{.*}} = arith.addf %[[ARG0_SHAPED]], %[[ARG0_SHAPED]]
    // CHECK: %{{.*}} = arith.addf %[[ARG1_SHAPED]], %[[ARG1_SHAPED]]
    // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<4x4xf32> to vector<16xf32>
    // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<2x2x2xf16> to vector<8xf16>
    // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<16xf32>, vector<8xf16>
    scf.yield %r0, %r1 : vector<4x4xf32>, vector<2x2x2xf16>
  }
  
  return %result#0, %result#1 : vector<4x4xf32>, vector<2x2x2xf16>
}

// Test case 5: Already flattened vector (1D) - may still have identity shape_casts
// CHECK-LABEL: @already_flat_vector
func.func @already_flat_vector(%v0: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c4 step %c1 
      iter_args(%arg0 = %v0) 
      -> (vector<64xi32>) {
    %temp = arith.addi %arg0, %arg0 : vector<64xi32>
    // CHECK: arith.addi
    // CHECK: scf.yield %{{.*}} : vector<64xi32>
    scf.yield %temp : vector<64xi32>
  }
  
  return %result : vector<64xi32>
}

// Test case 6: Nested loop with vector iter_args
// CHECK-LABEL: @flatten_nested_loops
func.func @flatten_nested_loops(%v0: vector<2x2xi16>) -> vector<2x2xi16> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[V0_FLAT:.*]] = vector.shape_cast %{{.*}} : vector<2x2xi16> to vector<4xi16>
  // CHECK: %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[V0_FLAT]]) -> (vector<4xi16>)
  %outer = scf.for %i = %c0 to %c2 step %c1 
      iter_args(%arg_outer = %v0) 
      -> (vector<2x2xi16>) {
    // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<4xi16> to vector<2x2xi16>
    
    // Inner loop should also be flattened when the transform is applied to it
    %inner = scf.for %j = %c0 to %c2 step %c1 
        iter_args(%arg_inner = %arg_outer) 
        -> (vector<2x2xi16>) {
      %temp = arith.addi %arg_inner, %arg_inner : vector<2x2xi16>
      scf.yield %temp : vector<2x2xi16>
    }
    
    // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<2x2xi16> to vector<4xi16>
    // CHECK: scf.yield %{{.*}} : vector<4xi16>
    scf.yield %inner : vector<2x2xi16>
  }
  
  return %outer : vector<2x2xi16>
}
