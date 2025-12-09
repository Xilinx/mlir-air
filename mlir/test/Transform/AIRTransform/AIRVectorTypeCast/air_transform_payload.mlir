//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test case 1: Basic vector.fma operation cast from f32 to f16
// CHECK-LABEL: @vector_fma_f32_to_f16
func.func @vector_fma_f32_to_f16(%a: vector<8xf32>, %b: vector<8xf32>, %c: vector<8xf32>) -> vector<8xf32> {
  %result = vector.fma %a, %b, %c : vector<8xf32>
  
  // CHECK: %[[A_CAST:.*]] = arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>
  // CHECK: %[[B_CAST:.*]] = arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>
  // CHECK: %[[C_CAST:.*]] = arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>
  // CHECK: %[[RESULT_F16:.*]] = vector.fma %[[A_CAST]], %[[B_CAST]], %[[C_CAST]] : vector<8xf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_F16]] : vector<8xf16> to vector<8xf32>
  
  return %result : vector<8xf32>
}

// Test case 2: Vector addition with bf16 target type
// CHECK-LABEL: @vector_add_f32_to_bf16
func.func @vector_add_f32_to_bf16(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %result = arith.addf %a, %b : vector<16xf32>
  
  // CHECK: %[[A_CAST:.*]] = arith.truncf %{{.*}} : vector<16xf32> to vector<16xbf16>
  // CHECK: %[[B_CAST:.*]] = arith.truncf %{{.*}} : vector<16xf32> to vector<16xbf16>
  // CHECK: %[[RESULT_BF16:.*]] = arith.addf %[[A_CAST]], %[[B_CAST]] : vector<16xbf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_BF16]] : vector<16xbf16> to vector<16xf32>
  
  return %result : vector<16xf32>
}

// Test case 3: Vector multiplication with 2D vectors
// CHECK-LABEL: @vector_mul_2d_f32_to_f16
func.func @vector_mul_2d_f32_to_f16(%a: vector<4x8xf32>, %b: vector<4x8xf32>) -> vector<4x8xf32> {
  %result = arith.mulf %a, %b : vector<4x8xf32>
  
  // CHECK: %[[A_CAST:.*]] = arith.truncf %{{.*}} : vector<4x8xf32> to vector<4x8xf16>
  // CHECK: %[[B_CAST:.*]] = arith.truncf %{{.*}} : vector<4x8xf32> to vector<4x8xf16>
  // CHECK: %[[RESULT_F16:.*]] = arith.mulf %[[A_CAST]], %[[B_CAST]] : vector<4x8xf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_F16]] : vector<4x8xf16> to vector<4x8xf32>
  
  return %result : vector<4x8xf32>
}

// Test case 4: Vector contract operation (matrix multiplication)
// CHECK-LABEL: @vector_contract_f32_to_f16
func.func @vector_contract_f32_to_f16(%lhs: vector<4x8xf32>, %rhs: vector<8x4xf32>, %acc: vector<4x4xf32>) -> vector<4x4xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %acc : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  
  // CHECK: %[[LHS_CAST:.*]] = arith.truncf %{{.*}} : vector<4x8xf32> to vector<4x8xf16>
  // CHECK: %[[RHS_CAST:.*]] = arith.truncf %{{.*}} : vector<8x4xf32> to vector<8x4xf16>
  // CHECK: %[[ACC_CAST:.*]] = arith.truncf %{{.*}} : vector<4x4xf32> to vector<4x4xf16>
  // CHECK: %[[RESULT_F16:.*]] = vector.contract
  // CHECK-SAME: %[[LHS_CAST]], %[[RHS_CAST]], %[[ACC_CAST]] : vector<4x8xf16>, vector<8x4xf16> into vector<4x4xf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_F16]] : vector<4x4xf16> to vector<4x4xf32>
  
  return %result : vector<4x4xf32>
}

// Test case 5: Vector reduction operation
// CHECK-LABEL: @vector_reduction_f32_to_f16
func.func @vector_reduction_f32_to_f16(%input: vector<16xf32>) -> f32 {
  %result = vector.reduction <add>, %input : vector<16xf32> into f32
  
  // CHECK: %[[INPUT_CAST:.*]] = arith.truncf %{{.*}} : vector<16xf32> to vector<16xf16>
  // CHECK: %[[RESULT_F16:.*]] = vector.reduction <add>, %[[INPUT_CAST]] : vector<16xf16> into f16
  // CHECK: %{{.*}} = arith.extf %[[RESULT_F16]] : f16 to f32
  
  return %result : f32
}

// Test case 7: Integer vector operations (i32 to i16)
// CHECK-LABEL: @vector_int_i32_to_i16
func.func @vector_int_i32_to_i16(%a: vector<8xi32>, %b: vector<8xi32>) -> vector<8xi32> {
  %result = arith.addi %a, %b : vector<8xi32>
  
  // CHECK: %[[A_CAST:.*]] = arith.trunci %{{.*}} : vector<8xi32> to vector<8xi16>
  // CHECK: %[[B_CAST:.*]] = arith.trunci %{{.*}} : vector<8xi32> to vector<8xi16>
  // CHECK: %[[RESULT_I16:.*]] = arith.addi %[[A_CAST]], %[[B_CAST]] : vector<8xi16>
  // CHECK: %{{.*}} = arith.extsi %[[RESULT_I16]] : vector<8xi16> to vector<8xi32>
  
  return %result : vector<8xi32>
}

// Test case 8: Multiple vector operations in sequence
// CHECK-LABEL: @vector_sequence_f32_to_f16
func.func @vector_sequence_f32_to_f16(%a: vector<8xf32>, %b: vector<8xf32>, %c: vector<8xf32>) -> vector<8xf32> {
  %temp = arith.addf %a, %b : vector<8xf32>
  %result = arith.mulf %temp, %c : vector<8xf32>
  
  // CHECK: %[[A_CAST:.*]] = arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>
  // CHECK: %[[B_CAST:.*]] = arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>
  // CHECK: %[[TEMP_F16:.*]] = arith.addf %[[A_CAST]], %[[B_CAST]] : vector<8xf16>
  // CHECK: %[[TEMP:.*]] = arith.extf %[[TEMP_F16]] : vector<8xf16> to vector<8xf32>
  // CHECK: %[[TEMP_CAST:.*]] = arith.truncf %[[TEMP]] : vector<8xf32> to vector<8xf16>
  // CHECK: %[[C_CAST:.*]] = arith.truncf %{{.*}} : vector<8xf32> to vector<8xf16>
  // CHECK: %[[RESULT_F16:.*]] = arith.mulf %[[TEMP_CAST]], %[[C_CAST]] : vector<8xf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_F16]] : vector<8xf16> to vector<8xf32>
  
  return %result : vector<8xf32>
}

// Test case 9: All single-element vectors should NOT be cast (new feature)
// CHECK-LABEL: @single_element_input_not_cast
func.func @single_element_input_not_cast(%a: vector<1xf32>, %b: vector<1xf32>) -> vector<1xf32> {
  // When ALL vectors are single-element, skip casting entirely
  %result = arith.addf %a, %b : vector<1xf32>
  
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: arith.extf
  // CHECK: %[[RESULT:.*]] = arith.addf %{{.*}}, %{{.*}} : vector<1xf32>
  
  return %result : vector<1xf32>
}

// Test case 10: vector.multi_reduction with mixed vector sizes (new feature)
// CHECK-LABEL: @multi_reduction_single_output
func.func @multi_reduction_single_output(%input: vector<1x16xf32>, %acc: vector<1xf32>) -> vector<1xf32> {
  %result = vector.multi_reduction <add>, %input, %acc [1] : vector<1x16xf32> to vector<1xf32>
  
  // Since NOT ALL vectors are single-element (input has 16 elements), all will be cast
  // CHECK: %[[INPUT_CAST:.*]] = arith.truncf %{{.*}} : vector<1x16xf32> to vector<1x16xf16>
  // CHECK: %[[ACC_CAST:.*]] = arith.truncf %{{.*}} : vector<1xf32> to vector<1xf16>
  // CHECK: %[[RESULT_F16:.*]] = vector.multi_reduction <add>, %[[INPUT_CAST]], %[[ACC_CAST]] [1] : vector<1x16xf16> to vector<1xf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_F16]] : vector<1xf16> to vector<1xf32>
  
  return %result : vector<1xf32>
}
