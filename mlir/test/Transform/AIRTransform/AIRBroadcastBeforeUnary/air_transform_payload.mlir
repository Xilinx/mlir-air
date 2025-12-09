//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test basic pattern: broadcast before rsqrt
// CHECK-LABEL: @test_basic_broadcast_before_rsqrt
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %arg0 : vector<1xf32> to vector<16xf32>
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<16xf32>
// CHECK-NEXT: return %[[RSQRT]]
func.func @test_basic_broadcast_before_rsqrt(%arg0: vector<1xf32>) -> vector<16xf32> {
  %rsqrt = math.rsqrt %arg0 : vector<1xf32>
  %result = vector.broadcast %rsqrt : vector<1xf32> to vector<16xf32>
  return %result : vector<16xf32>
}

// -----

// Test with actual layernorm variance computation pattern
// CHECK-LABEL: @test_layernorm_pattern
// CHECK: %[[VAR_EPS:.*]] = arith.addf
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[VAR_EPS]] : vector<1xf32> to vector<16xf32>
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<16xf32>
// CHECK: return %[[RSQRT]]
func.func @test_layernorm_pattern(%arg0: vector<1xf32>, %arg1: vector<1xf32>) -> vector<16xf32> {
  %cst_eps = arith.constant dense<9.99999974E-6> : vector<1xf32>
  %cst_n = arith.constant dense<1.024000e+03> : vector<1xf32>
  
  %mean = arith.divf %arg0, %cst_n : vector<1xf32>
  %mean_sq = arith.mulf %mean, %mean : vector<1xf32>
  %mean_sq_full = arith.divf %arg1, %cst_n : vector<1xf32>
  %var = arith.subf %mean_sq_full, %mean_sq : vector<1xf32>
  %var_eps = arith.addf %var, %cst_eps : vector<1xf32>
  
  %rsqrt = math.rsqrt %var_eps : vector<1xf32>
  %broadcast = vector.broadcast %rsqrt : vector<1xf32> to vector<16xf32>
  
  return %broadcast : vector<16xf32>
}

// -----

// Test that multi-element vectors are not transformed
// CHECK-LABEL: @test_no_transform_multi_element
// CHECK: %[[RSQRT:.*]] = math.rsqrt %arg0 : vector<4xf32>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.broadcast %[[RSQRT]] : vector<4xf32> to vector<4x4xf32>
// CHECK-NEXT: return %[[BROADCAST]]
func.func @test_no_transform_multi_element(%arg0: vector<4xf32>) -> vector<4x4xf32> {
  %rsqrt = math.rsqrt %arg0 : vector<4xf32>
  %result = vector.broadcast %rsqrt : vector<4xf32> to vector<4x4xf32>
  return %result : vector<4x4xf32>
}

// -----

// Test with bf16 type
// CHECK-LABEL: @test_broadcast_before_rsqrt_bf16
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %arg0 : vector<1xbf16> to vector<16xbf16>
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<16xbf16>
// CHECK-NEXT: return %[[RSQRT]]
func.func @test_broadcast_before_rsqrt_bf16(%arg0: vector<1xbf16>) -> vector<16xbf16> {
  %rsqrt = math.rsqrt %arg0 : vector<1xbf16>
  %result = vector.broadcast %rsqrt : vector<1xbf16> to vector<16xbf16>
  return %result : vector<16xbf16>
}

// -----

// Test multiple transformations in same function
// CHECK-LABEL: @test_multiple_patterns
// CHECK: %[[BROADCAST1:.*]] = vector.broadcast %arg0 : vector<1xf32> to vector<16xf32>
// CHECK-NEXT: %[[RSQRT1:.*]] = math.rsqrt %[[BROADCAST1]] : vector<16xf32>
// CHECK: %[[BROADCAST2:.*]] = vector.broadcast %arg1 : vector<1xf32> to vector<8xf32>
// CHECK-NEXT: %[[RSQRT2:.*]] = math.rsqrt %[[BROADCAST2]] : vector<8xf32>
// CHECK: return %[[RSQRT1]], %[[RSQRT2]]
func.func @test_multiple_patterns(%arg0: vector<1xf32>, %arg1: vector<1xf32>) -> (vector<16xf32>, vector<8xf32>) {
  %rsqrt1 = math.rsqrt %arg0 : vector<1xf32>
  %result1 = vector.broadcast %rsqrt1 : vector<1xf32> to vector<16xf32>
  
  %rsqrt2 = math.rsqrt %arg1 : vector<1xf32>
  %result2 = vector.broadcast %rsqrt2 : vector<1xf32> to vector<8xf32>
  
  return %result1, %result2 : vector<16xf32>, vector<8xf32>
}

// -----

// Test that rsqrt with multiple uses is not transformed
// CHECK-LABEL: @test_no_transform_multiple_uses
// CHECK: %[[RSQRT:.*]] = math.rsqrt %arg0 : vector<1xf32>
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[RSQRT]] : vector<1xf32> to vector<16xf32>
// CHECK: return %[[BROADCAST]], %[[RSQRT]]
func.func @test_no_transform_multiple_uses(%arg0: vector<1xf32>) -> (vector<16xf32>, vector<1xf32>) {
  %rsqrt = math.rsqrt %arg0 : vector<1xf32>
  %broadcast = vector.broadcast %rsqrt : vector<1xf32> to vector<16xf32>
  return %broadcast, %rsqrt : vector<16xf32>, vector<1xf32>
}

// -----

// Test 2D vector broadcast - shape preservation
// CHECK-LABEL: @test_2d_vector_shape_preservation
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %arg0 : vector<1xf32> to vector<1x16xf32>
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<1x16xf32>
// CHECK-NEXT: return %[[RSQRT]]
func.func @test_2d_vector_shape_preservation(%arg0: vector<1xf32>) -> vector<1x16xf32> {
  %rsqrt = math.rsqrt %arg0 : vector<1xf32>
  %result = vector.broadcast %rsqrt : vector<1xf32> to vector<1x16xf32>
  return %result : vector<1x16xf32>
}

// -----

// Test within memref operations (typical layernorm scenario)
// CHECK-LABEL: @test_layernorm_with_memref
// CHECK: %[[VAR_EPS:.*]] = arith.addf
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[VAR_EPS]] : vector<1xf32> to vector<16xf32>
// CHECK: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<16xf32>
// CHECK: vector.transfer_write %[[RSQRT]]
func.func @test_layernorm_with_memref(%arg0: memref<1xf32, 2>, %arg1: memref<1xf32, 2>, %arg2: memref<1x16xf32, 2>) {
  %c0 = arith.constant 0 : index
  %cst_eps = arith.constant dense<9.99999974E-6> : vector<1xf32>
  %cst_n = arith.constant dense<1.024000e+03> : vector<1xf32>
  %pad = arith.constant 0.0 : f32
  
  %mean_vec = vector.transfer_read %arg0[%c0], %pad {in_bounds = [true]} : memref<1xf32, 2>, vector<1xf32>
  %var_vec = vector.transfer_read %arg1[%c0], %pad {in_bounds = [true]} : memref<1xf32, 2>, vector<1xf32>
  
  %mean = arith.divf %mean_vec, %cst_n : vector<1xf32>
  %mean_sq = arith.mulf %mean, %mean : vector<1xf32>
  %var_scaled = arith.divf %var_vec, %cst_n : vector<1xf32>
  %var = arith.subf %var_scaled, %mean_sq : vector<1xf32>
  %var_eps = arith.addf %var, %cst_eps : vector<1xf32>
  
  %rsqrt = math.rsqrt %var_eps : vector<1xf32>
  %broadcast = vector.broadcast %rsqrt : vector<1xf32> to vector<16xf32>
  
  vector.transfer_write %broadcast, %arg2[%c0, %c0] {in_bounds = [true]} : vector<16xf32>, memref<1x16xf32, 2>
  return
}
