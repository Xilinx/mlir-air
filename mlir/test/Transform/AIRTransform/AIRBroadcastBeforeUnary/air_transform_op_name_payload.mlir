//===- air_transform_op_name_payload.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform_op_name.mlir' %s | FileCheck %s

// Test scalar math.rsqrt with op_name filter (the exact scenario from issue #1394)
// CHECK-LABEL: @test_scalar_rsqrt_op_name
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %arg0 : f32 to vector<16xf32>
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<16xf32>
// CHECK-NEXT: return %[[RSQRT]]
func.func @test_scalar_rsqrt_op_name(%arg0: f32) -> vector<16xf32> {
  %rsqrt = math.rsqrt %arg0 : f32
  %result = vector.broadcast %rsqrt : f32 to vector<16xf32>
  return %result : vector<16xf32>
}

// -----

// Test that op_name filter only matches math.rsqrt, not other unary ops
// arith.negf should NOT be transformed when op_name = "math.rsqrt"
// CHECK-LABEL: @test_scalar_negf_not_matched_by_op_name
// CHECK: %[[NEGF:.*]] = arith.negf %arg0 : f32
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.broadcast %[[NEGF]] : f32 to vector<16xf32>
// CHECK-NEXT: return %[[BROADCAST]]
func.func @test_scalar_negf_not_matched_by_op_name(%arg0: f32) -> vector<16xf32> {
  %neg = arith.negf %arg0 : f32
  %result = vector.broadcast %neg : f32 to vector<16xf32>
  return %result : vector<16xf32>
}

// -----

// Test vector<1xf32> rsqrt still works with op_name filter
// CHECK-LABEL: @test_vector1_rsqrt_op_name
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %arg0 : vector<1xf32> to vector<16xf32>
// CHECK-NEXT: %[[RSQRT:.*]] = math.rsqrt %[[BROADCAST]] : vector<16xf32>
// CHECK-NEXT: return %[[RSQRT]]
func.func @test_vector1_rsqrt_op_name(%arg0: vector<1xf32>) -> vector<16xf32> {
  %rsqrt = math.rsqrt %arg0 : vector<1xf32>
  %result = vector.broadcast %rsqrt : vector<1xf32> to vector<16xf32>
  return %result : vector<16xf32>
}
