//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test case 1: Fuse extf + mulf with a reduction operation
// CHECK-LABEL: @fuse_extf_mulf_with_reduce
func.func @fuse_extf_mulf_with_reduce(%input: tensor<16x8xf16>, %init_reduce: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<16x8xf32>
  
  // First op: extf + mulf (element-wise, parallel iterations)
  %transformed = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<16x8xf16>) outs(%empty : tensor<16x8xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.mulf %0, %cst : f32
    linalg.yield %1 : f32
  } -> tensor<16x8xf32>
  
  // Second op: reduction (consumes first op's result)
  // CHECK: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<16x8xf16>) outs(%arg1 : tensor<16xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: f16, %[[ACC:.+]]: f32):
  // CHECK-NEXT: %[[EXT:.+]] = arith.extf %[[IN]] : f16 to f32
  // CHECK-NEXT: %[[MUL:.+]] = arith.mulf %[[EXT]]
  // CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[MUL]], %[[ACC]]
  // CHECK-NEXT: linalg.yield %[[ADD]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%transformed : tensor<16x8xf32>) outs(%init_reduce : tensor<16xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %2 = arith.addf %in, %acc : f32
    linalg.yield %2 : f32
  } -> tensor<16xf32>
  
  return %result : tensor<16xf32>
}

// Test case 2: Fuse single extf (backward compatibility test)
// CHECK-LABEL: @fuse_single_extf
func.func @fuse_single_extf(%input: tensor<16xf16>, %other: tensor<16xf32>) -> tensor<16xf32> {
  %empty1 = tensor.empty() : tensor<16xf32>
  %empty2 = tensor.empty() : tensor<16xf32>
  
  // First op: only extf
  %extf_result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<16xf16>) outs(%empty1 : tensor<16xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    linalg.yield %0 : f32
  } -> tensor<16xf32>
  
  // Second op: addf
  // CHECK: linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<16xf16>, tensor<16xf32>) outs(%1 : tensor<16xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: f16, %[[OTHER:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK-NEXT: %[[EXT:.+]] = arith.extf %[[IN]] : f16 to f32
  // CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[EXT]], %[[OTHER]]
  // CHECK-NEXT: linalg.yield %[[ADD]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%extf_result, %other : tensor<16xf32>, tensor<16xf32>) outs(%empty2 : tensor<16xf32>) {
  ^bb0(%in: f32, %other_val: f32, %out: f32):
    %1 = arith.addf %in, %other_val : f32
    linalg.yield %1 : f32
  } -> tensor<16xf32>
  
  return %result : tensor<16xf32>
}

// Test case 3: Fuse with math operations (extf + sqrt + mulf)
// CHECK-LABEL: @fuse_with_math_ops
func.func @fuse_with_math_ops(%input: tensor<8xf16>, %other: tensor<8xf32>) -> tensor<8xf32> {
  %cst = arith.constant 3.0 : f32
  %empty1 = tensor.empty() : tensor<8xf32>
  %empty2 = tensor.empty() : tensor<8xf32>
  
  // First op: extf + sqrt + mulf
  %transformed = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<8xf16>) outs(%empty1 : tensor<8xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = math.sqrt %0 : f32
    %2 = arith.mulf %1, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<8xf32>
  
  // Second op: addf
  // CHECK: linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<8xf16>, tensor<8xf32>) outs(%1 : tensor<8xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: f16, %[[OTHER:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK-NEXT: %[[EXT:.+]] = arith.extf %[[IN]] : f16 to f32
  // CHECK-NEXT: %[[SQRT:.+]] = math.sqrt %[[EXT]]
  // CHECK-NEXT: %[[MUL:.+]] = arith.mulf %[[SQRT]]
  // CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[MUL]], %[[OTHER]]
  // CHECK-NEXT: linalg.yield %[[ADD]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%transformed, %other : tensor<8xf32>, tensor<8xf32>) outs(%empty2 : tensor<8xf32>) {
  ^bb0(%in: f32, %other_val: f32, %out: f32):
    %3 = arith.addf %in, %other_val : f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  
  return %result : tensor<8xf32>
}

// Test case 4: 2D tensors with multiple operations
// CHECK-LABEL: @fuse_2d_multi_ops
func.func @fuse_2d_multi_ops(%input: tensor<4x8xf16>, %other: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst_add = arith.constant 1.0 : f32
  %cst_mul = arith.constant 2.0 : f32
  %empty1 = tensor.empty() : tensor<4x8xf32>
  %empty2 = tensor.empty() : tensor<4x8xf32>
  
  // First op: extf + addf + mulf (chain of operations)
  %transformed = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x8xf16>) outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.addf %0, %cst_add : f32
    %2 = arith.mulf %1, %cst_mul : f32
    linalg.yield %2 : f32
  } -> tensor<4x8xf32>
  
  // Second op: subf
  // CHECK: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf32>) outs(%1 : tensor<4x8xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: f16, %[[OTHER:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK-NEXT: %[[EXT:.+]] = arith.extf %[[IN]] : f16 to f32
  // CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[EXT]]
  // CHECK-NEXT: %[[MUL:.+]] = arith.mulf %[[ADD]]
  // CHECK-NEXT: %[[SUB:.+]] = arith.subf %[[MUL]], %[[OTHER]]
  // CHECK-NEXT: linalg.yield %[[SUB]]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%transformed, %other : tensor<4x8xf32>, tensor<4x8xf32>) outs(%empty2 : tensor<4x8xf32>) {
  ^bb0(%in: f32, %other_val: f32, %out: f32):
    %3 = arith.subf %in, %other_val : f32
    linalg.yield %3 : f32
  } -> tensor<4x8xf32>
  
  return %result : tensor<4x8xf32>
}

// Test case 5: Multi-input first op with reduction (real-world softmax pattern)
// CHECK-LABEL: @fuse_multi_input_with_reduce
func.func @fuse_multi_input_with_reduce(%input: tensor<4x256xbf16>, %max_vals: tensor<4xf32>) -> tensor<4xf32> {
  %cst_0 = arith.constant 0.0 : f32
  %empty1 = tensor.empty() : tensor<4x256xf32>
  %empty2 = tensor.empty() : tensor<4xf32>
  
  // First op: extf + subf + exp (with 2 inputs)
  %transformed = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %max_vals : tensor<4x256xbf16>, tensor<4xf32>) outs(%empty1 : tensor<4x256xf32>) {
  ^bb0(%in: bf16, %in_max: f32, %out: f32):
    %0 = arith.extf %in : bf16 to f32
    %1 = arith.subf %0, %in_max : f32
    %2 = math.exp %1 : f32
    linalg.yield %2 : f32
  } -> tensor<4x256xf32>
  
  // Second op: reduction sum
  // CHECK: linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x256xbf16>, tensor<4xf32>) outs(%3 : tensor<4xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: bf16, %[[MAX:.+]]: f32, %[[ACC:.+]]: f32):
  // CHECK-NEXT: %[[EXT:.+]] = arith.extf %[[IN]] : bf16 to f32
  // CHECK-NEXT: %[[SUB:.+]] = arith.subf %[[EXT]], %[[MAX]]
  // CHECK-NEXT: %[[EXP:.+]] = math.exp %[[SUB]]
  // CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[EXP]], %[[ACC]]
  // CHECK-NEXT: linalg.yield %[[ADD]]
  %init = linalg.fill ins(%cst_0 : f32) outs(%empty2 : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%transformed : tensor<4x256xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %3 = arith.addf %in, %acc : f32
    linalg.yield %3 : f32
  } -> tensor<4xf32>
  
  return %result : tensor<4xf32>
}
