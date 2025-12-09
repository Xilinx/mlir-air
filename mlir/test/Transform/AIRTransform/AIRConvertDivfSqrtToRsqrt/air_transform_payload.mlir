//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case for ConvertDivfSqrtToRsqrtOp transform
// This demonstrates the conversion of divf(1.0, sqrt(x)) pattern to rsqrt(x)

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @convert_divf_sqrt_to_rsqrt_vector
func.func @convert_divf_sqrt_to_rsqrt_vector(%arg0: memref<1x1024xf32, 2>) {
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant dense<1.000000e+00> : vector<1xf32>
  %cst_eps = arith.constant dense<9.99999974E-6> : vector<1xf32>
  %poison = ub.poison : f32
  
  // Read variance value
  %variance_raw = vector.transfer_read %arg0[%c0, %c0], %poison {in_bounds = [true]} : memref<1x1024xf32, 2>, vector<1xf32>
  
  // Add epsilon for numerical stability
  // CHECK: %[[VAR:.*]] = arith.addf
  %variance = arith.addf %variance_raw, %cst_eps : vector<1xf32>
  
  // Pattern to be optimized: sqrt followed by division by 1.0
  // This should be converted to math.rsqrt
  // CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[VAR]] : vector<1xf32>
  // CHECK-NOT: math.sqrt
  // CHECK-NOT: arith.divf %cst{{.*}}, %{{.*}} : vector<1xf32>
  %sqrt_variance = math.sqrt %variance : vector<1xf32>
  %inv_std = arith.divf %cst_1, %sqrt_variance : vector<1xf32>
  
  // Write result back
  // CHECK: vector.transfer_write %[[RSQRT]]
  vector.transfer_write %inv_std, %arg0[%c0, %c0] {in_bounds = [true]} : vector<1xf32>, memref<1x1024xf32, 2>
  return
}

// CHECK-LABEL: @convert_divf_sqrt_to_rsqrt_scalar
func.func @convert_divf_sqrt_to_rsqrt_scalar(%arg0: f32) -> f32 {
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  
  // Add epsilon
  // CHECK: %[[VAR:.*]] = arith.addf
  %variance = arith.addf %arg0, %cst_eps : f32
  
  // Pattern to be optimized for scalar type
  // CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[VAR]] : f32
  %sqrt_variance = math.sqrt %variance : f32
  %inv_std = arith.divf %cst_1, %sqrt_variance : f32
  
  // CHECK: return %[[RSQRT]]
  return %inv_std : f32
}

// CHECK-LABEL: @layernorm_pattern_from_triton
func.func @layernorm_pattern_from_triton(%input: memref<1x1024xf32, 2>, %output: memref<1x1024xf32, 2>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %cst_1 = arith.constant dense<1.000000e+00> : vector<1xf32>
  %cst_eps = arith.constant dense<9.99999974E-6> : vector<1xf32>
  %cst_1024 = arith.constant dense<1.024000e+03> : vector<1xf32>
  %poison = ub.poison : f32
  
  // Compute mean
  %sum = vector.transfer_read %input[%c0, %c0], %poison {in_bounds = [true]} : memref<1x1024xf32, 2>, vector<1xf32>
  %mean = arith.divf %sum, %cst_1024 : vector<1xf32>
  
  // Compute variance
  %mean_squared = arith.mulf %mean, %mean : vector<1xf32>
  %sum_squares = vector.transfer_read %input[%c0, %c0], %poison {in_bounds = [true]} : memref<1x1024xf32, 2>, vector<1xf32>
  %mean_of_squares = arith.divf %sum_squares, %cst_1024 : vector<1xf32>
  %variance_raw = arith.subf %mean_of_squares, %mean_squared : vector<1xf32>
  // CHECK: %[[VAR:.*]] = arith.addf
  %variance = arith.addf %variance_raw, %cst_eps : vector<1xf32>
  
  // Compute inverse standard deviation - THIS IS THE PATTERN
  // CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[VAR]] : vector<1xf32>
  %sqrt_var = math.sqrt %variance : vector<1xf32>
  %inv_std = arith.divf %cst_1, %sqrt_var : vector<1xf32>
  
  // Normalize: (x - mean) * inv_std
  %data = vector.transfer_read %input[%c0, %c0], %poison {in_bounds = [true]} : memref<1x1024xf32, 2>, vector<16xf32>
  %mean_broadcast = vector.broadcast %mean : vector<1xf32> to vector<16xf32>
  %centered = arith.subf %data, %mean_broadcast : vector<16xf32>
  // CHECK: %[[BCAST:.*]] = vector.broadcast %[[RSQRT]]
  %inv_std_broadcast = vector.broadcast %inv_std : vector<1xf32> to vector<16xf32>
  %normalized = arith.mulf %centered, %inv_std_broadcast : vector<16xf32>
  %result = arith.addf %normalized, %cst : vector<16xf32>
  
  vector.transfer_write %result, %output[%c0, %c0] {in_bounds = [true]} : vector<16xf32>, memref<1x1024xf32, 2>
  return
}

// Negative test: divf with non-constant numerator should NOT be converted
// CHECK-LABEL: @no_convert_non_constant
func.func @no_convert_non_constant(%arg0: f32, %arg1: f32) -> f32 {
  %variance = arith.addf %arg0, %arg1 : f32
  
  // This should NOT be converted because numerator is not 1.0
  // CHECK: math.sqrt
  // CHECK: arith.divf
  // CHECK-NOT: math.rsqrt
  %sqrt_variance = math.sqrt %variance : f32
  %result = arith.divf %arg1, %sqrt_variance : f32
  
  return %result : f32
}

// Negative test: sqrt with multiple uses should NOT be converted
// CHECK-LABEL: @no_convert_multiple_uses
func.func @no_convert_multiple_uses(%arg0: f32) -> (f32, f32) {
  %cst_1 = arith.constant 1.000000e+00 : f32
  %cst_2 = arith.constant 2.000000e+00 : f32
  
  // sqrt has two uses, should NOT be converted
  // CHECK: %[[SQRT:.*]] = math.sqrt
  // CHECK: arith.divf %{{.*}}, %[[SQRT]]
  // CHECK: arith.mulf %[[SQRT]], %{{.*}}
  // CHECK-NOT: math.rsqrt
  %sqrt_val = math.sqrt %arg0 : f32
  %inv_sqrt = arith.divf %cst_1, %sqrt_val : f32
  %sqrt_times_2 = arith.mulf %sqrt_val, %cst_2 : f32
  
  return %inv_sqrt, %sqrt_times_2 : f32, f32
}
