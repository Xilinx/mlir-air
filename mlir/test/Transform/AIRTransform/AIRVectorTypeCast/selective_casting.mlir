//===- selective_casting.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/selective_casting_transform.mlir' %s | FileCheck %s

// Test 1: Default behavior - cast all inputs and outputs
// CHECK-LABEL: @contract_cast_all_default
func.func @contract_cast_all_default(%lhs: vector<4x8xf32>, %rhs: vector<8x4xf32>, %acc: vector<4x4xf32>) -> vector<4x4xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %acc : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  
  // All inputs and output should be cast
  // CHECK: %[[LHS_CAST:.*]] = arith.truncf %{{.*}} : vector<4x8xf32> to vector<4x8xbf16>
  // CHECK: %[[RHS_CAST:.*]] = arith.truncf %{{.*}} : vector<8x4xf32> to vector<8x4xbf16>
  // CHECK: %[[ACC_CAST:.*]] = arith.truncf %{{.*}} : vector<4x4xf32> to vector<4x4xbf16>
  // CHECK: %[[RESULT_BF16:.*]] = vector.contract
  // CHECK-SAME: %[[LHS_CAST]], %[[RHS_CAST]], %[[ACC_CAST]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xbf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_BF16]] : vector<4x4xbf16> to vector<4x4xf32>
  
  return %result : vector<4x4xf32>
}

// Test 2: Cast only lhs and rhs (inputs 0, 1) keeping acc and output as f32
// Since lhs+rhs must have same type, and acc+result must have same type,
// we cast lhs+rhs to bf16, cast acc+result to bf16 as well to satisfy constraints
// CHECK-LABEL: @contract_cast_lhs_rhs_with_acc_output
func.func @contract_cast_lhs_rhs_with_acc_output(%lhs: vector<4x8xf32>, %rhs: vector<8x4xf32>, %acc: vector<4x4xf32>) -> vector<4x4xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %acc : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  
  // With input_indices=[0,1], cast lhs and rhs to bf16
  // With empty output_indices (default), output also gets cast
  // Since acc and output must have same type, acc is also cast
  // This produces valid IR: lhs/rhs are bf16, acc/result are bf16
  // CHECK: %[[LHS_CAST:.*]] = arith.truncf %{{.*}} : vector<4x8xf32> to vector<4x8xbf16>
  // CHECK: %[[RHS_CAST:.*]] = arith.truncf %{{.*}} : vector<8x4xf32> to vector<8x4xbf16>
  // CHECK: %[[ACC_CAST:.*]] = arith.truncf %{{.*}} : vector<4x4xf32> to vector<4x4xbf16>
  // CHECK: %[[RESULT_BF16:.*]] = vector.contract
  // CHECK-SAME: %[[LHS_CAST]], %[[RHS_CAST]], %[[ACC_CAST]] : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xbf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_BF16]] : vector<4x4xbf16> to vector<4x4xf32>
  
  return %result : vector<4x4xf32>
}

// Test 3: Cast only acc and result (input 2, output 0) - keep lhs and rhs as f32
// This respects vector.contract constraint: lhs and rhs same type, acc and result same type
// CHECK-LABEL: @contract_cast_acc_result_only
func.func @contract_cast_acc_result_only(%lhs: vector<4x8xf32>, %rhs: vector<8x4xf32>, %acc: vector<4x4xf32>) -> vector<4x4xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %acc : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  
  // Only acc (input 2) and result (output 0) should be cast to bf16
  // lhs and rhs remain f32 (valid since lhs and rhs must have same type, acc and result must have same type)
  // CHECK: %[[ACC_CAST:.*]] = arith.truncf %{{.*}} : vector<4x4xf32> to vector<4x4xbf16>
  // CHECK: %[[RESULT_BF16:.*]] = vector.contract
  // CHECK-SAME: %arg0, %arg1, %[[ACC_CAST]] : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xbf16>
  // CHECK: %{{.*}} = arith.extf %[[RESULT_BF16]] : vector<4x4xbf16> to vector<4x4xf32>
  
  return %result : vector<4x4xf32>
}
