//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic fusion of extf with elementwise add
// CHECK-LABEL: @fuse_extf_with_add
func.func @fuse_extf_with_add(%input: tensor<16xf16>, %other: tensor<16xf32>) -> tensor<16xf32> {
  %empty1 = tensor.empty() : tensor<16xf32>
  %empty2 = tensor.empty() : tensor<16xf32>
  
  // First op: contains only arith.extf
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<16xf16>) outs(%empty1 : tensor<16xf32>) {
  ^bb0(%arg0: f16, %arg1: f32):
    %0 = arith.extf %arg0 : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // Second op: consumes the result of first op
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%extf_result, %other : tensor<16xf32>, tensor<16xf32>) outs(%empty2 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<16xf16>, tensor<16xf32>)
  // CHECK: ^bb0(%{{.*}}: f16, %{{.*}}: f32, %{{.*}}: f32):
  // CHECK:   %{{.*}} = arith.extf %{{.*}} : f16 to f32
  // CHECK:   %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK:   linalg.yield %{{.*}} : f32
  
  return %result : tensor<16xf32>
}

// Test case 2: Fusion with multiple inputs where extf result is not the first input
// CHECK-LABEL: @fuse_extf_with_mul_second_input
func.func @fuse_extf_with_mul_second_input(%input: tensor<8xf16>, %other: tensor<8xf32>) -> tensor<8xf32> {
  %empty1 = tensor.empty() : tensor<8xf32>
  %empty2 = tensor.empty() : tensor<8xf32>
  
  // First op: contains only arith.extf
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<8xf16>) outs(%empty1 : tensor<8xf32>) {
  ^bb0(%arg0: f16, %arg1: f32):
    %0 = arith.extf %arg0 : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  
  // Second op: extf_result is the second input
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%other, %extf_result : tensor<8xf32>, tensor<8xf32>) outs(%empty2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.mulf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  
  // CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf16>)
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f16, %{{.*}}: f32):
  // CHECK:   %{{.*}} = arith.extf %{{.*}} : f16 to f32
  // CHECK:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK:   linalg.yield %{{.*}} : f32
  
  return %result : tensor<8xf32>
}

// Test case 3: 2D tensor fusion
// CHECK-LABEL: @fuse_extf_2d_tensor
func.func @fuse_extf_2d_tensor(%input: tensor<4x8xf16>, %other: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty1 = tensor.empty() : tensor<4x8xf32>
  %empty2 = tensor.empty() : tensor<4x8xf32>
  
  // First op: 2D extf operation
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x8xf16>) outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%arg0: f16, %arg1: f32):
    %0 = arith.extf %arg0 : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  
  // Second op: 2D elementwise operation
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%extf_result, %other : tensor<4x8xf32>, tensor<4x8xf32>) outs(%empty2 : tensor<4x8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.subf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  
  // CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<4x8xf16>, tensor<4x8xf32>)
  // CHECK: ^bb0(%{{.*}}: f16, %{{.*}}: f32, %{{.*}}: f32):
  // CHECK:   %{{.*}} = arith.extf %{{.*}} : f16 to f32
  // CHECK:   %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f32
  // CHECK:   linalg.yield %{{.*}} : f32
  
  return %result : tensor<4x8xf32>
}

// Test case 4: Different precision extension (bf16 to f32)
// CHECK-LABEL: @fuse_extf_bf16_to_f32
func.func @fuse_extf_bf16_to_f32(%input: tensor<16xbf16>, %other: tensor<16xf32>) -> tensor<16xf32> {
  %empty1 = tensor.empty() : tensor<16xf32>
  %empty2 = tensor.empty() : tensor<16xf32>
  
  // First op: bf16 to f32 extension
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<16xbf16>) outs(%empty1 : tensor<16xf32>) {
  ^bb0(%arg0: bf16, %arg1: f32):
    %0 = arith.extf %arg0 : bf16 to f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // Second op: max operation
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%extf_result, %other : tensor<16xf32>, tensor<16xf32>) outs(%empty2 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : tensor<16xbf16>, tensor<16xf32>)
  // CHECK: ^bb0(%{{.*}}: bf16, %{{.*}}: f32, %{{.*}}: f32):
  // CHECK:   %{{.*}} = arith.extf %{{.*}} : bf16 to f32
  // CHECK:   %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : f32
  // CHECK:   linalg.yield %{{.*}} : f32
  
  return %result : tensor<16xf32>
}

// Test case 5: Negative test - first op contains more than just extf (should not fuse)
// CHECK-LABEL: @no_fuse_extf_with_extra_ops
func.func @no_fuse_extf_with_extra_ops(%input: tensor<16xf16>, %other: tensor<16xf32>) -> tensor<16xf32> {
  %empty1 = tensor.empty() : tensor<16xf32>
  %empty2 = tensor.empty() : tensor<16xf32>
  %cst = arith.constant 1.0 : f32
  
  // First op: contains extf AND additional operation (should not be fusable)
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<16xf16>) outs(%empty1 : tensor<16xf32>) {
  ^bb0(%arg0: f16, %arg1: f32):
    %0 = arith.extf %arg0 : f16 to f32
    %1 = arith.addf %0, %cst : f32
    linalg.yield %1 : f32
  } -> tensor<16xf32>
  
  // Second op: would consume the result
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%extf_result, %other : tensor<16xf32>, tensor<16xf32>) outs(%empty2 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.mulf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}} : tensor<16xf16>)
  // CHECK: arith.extf
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<16xf32>, tensor<16xf32>)
  
  return %result : tensor<16xf32>
}

// Test case 6: Negative test - operations are not directly connected (should not fuse)
// CHECK-LABEL: @no_fuse_not_directly_connected
func.func @no_fuse_not_directly_connected(%input: tensor<16xf16>, %other: tensor<16xf32>) -> tensor<16xf32> {
  %empty1 = tensor.empty() : tensor<16xf32>
  %empty2 = tensor.empty() : tensor<16xf32>
  
  // First op: contains only extf
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<16xf16>) outs(%empty1 : tensor<16xf32>) {
  ^bb0(%arg0: f16, %arg1: f32):
    %0 = arith.extf %arg0 : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // Second op: does NOT use extf_result (uses %other twice instead)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%other, %other : tensor<16xf32>, tensor<16xf32>) outs(%empty2 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}} : tensor<16xf16>)
  // CHECK: arith.extf
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<16xf32>, tensor<16xf32>)
  
  return %result : tensor<16xf32>
}
