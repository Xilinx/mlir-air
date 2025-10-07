//===- fuse_truncf.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/fuse_truncf_transform.mlir' %s | FileCheck %s

// Test 1: Fuse truncf into simple add producer
// CHECK-LABEL: @fuse_truncf_into_add
func.func @fuse_truncf_into_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %out: tensor<16xbf16>) -> tensor<16xbf16> {
  %empty = tensor.empty() : tensor<16xf32>
  
  // Producer operation: adds two f32 tensors
  %producer = linalg.generic {
    producer_op,
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) 
    outs(%empty : tensor<16xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %sum = arith.addf %in0, %in1 : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>
  
  // Truncf operation: truncates f32 to bf16
  %truncf = linalg.generic {
    truncf_op,
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%producer : tensor<16xf32>) 
    outs(%out : tensor<16xbf16>) {
  ^bb0(%in: f32, %out_elem: bf16):
    %trunc = arith.truncf %in : f32 to bf16
    linalg.yield %trunc : bf16
  } -> tensor<16xbf16>
  
  return %truncf : tensor<16xbf16>
}

// CHECK-NOT: linalg.generic {{.*}} attributes {truncf_op}
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME: outs(%arg2 : tensor<16xbf16>)
// CHECK: ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT_ELEM:.*]]: bf16):
// CHECK:   %[[SUM:.*]] = arith.addf %[[IN0]], %[[IN1]] : f32
// CHECK:   %[[TRUNC:.*]] = arith.truncf %[[SUM]] : f32 to bf16
// CHECK:   linalg.yield %[[TRUNC]] : bf16
// CHECK: return %[[RESULT]] : tensor<16xbf16>

// Test 2: Fuse truncf into matmul producer
// CHECK-LABEL: @fuse_truncf_into_matmul
func.func @fuse_truncf_into_matmul(%a: tensor<8x16xf32>, %b: tensor<16x8xf32>, %c: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
  %empty = tensor.empty() : tensor<8x8xf32>
  
  // Producer matmul in f32
  %matmul = linalg.matmul
    ins(%a, %b : tensor<8x16xf32>, tensor<16x8xf32>)
    outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  
  // Truncf operation: truncates f32 to bf16
  %truncf = linalg.generic {truncf_matmul,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul : tensor<8x8xf32>) 
    outs(%c : tensor<8x8xbf16>) {
  ^bb0(%in: f32, %out_elem: bf16):
    %trunc = arith.truncf %in : f32 to bf16
    linalg.yield %trunc : bf16
  } -> tensor<8x8xbf16>
  
  return %truncf : tensor<8x8xbf16>
}

// CHECK-NOT: linalg.generic {{.*}} attributes {truncf_matmul}
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x8xf32>)
// CHECK-SAME: outs(%arg2 : tensor<8x8xbf16>)
// CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %[[ACC:.*]]: bf16):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:   %[[TRUNC:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK:   linalg.yield %[[TRUNC]] : bf16
// CHECK: return %[[RESULT]] : tensor<8x8xbf16>

// Test 3: Fuse truncf (f32->bf16) into mul producer
// CHECK-LABEL: @fuse_truncf_f32_to_bf16
func.func @fuse_truncf_f32_to_bf16(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %out: tensor<32xbf16>) -> tensor<32xbf16> {
  %empty = tensor.empty() : tensor<32xf32>
  
  // Producer operation: multiplies two f32 tensors
  %producer = linalg.generic {
    producer_mul,
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) 
    outs(%empty : tensor<32xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
    %mul = arith.mulf %in0, %in1 : f32
    linalg.yield %mul : f32
  } -> tensor<32xf32>
  
  // Truncf operation: truncates f32 to bf16
  %truncf = linalg.generic {
    truncf_bf16,
    indexing_maps = [affine_map<(d0) -> (d0)>, 
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%producer : tensor<32xf32>) 
    outs(%out : tensor<32xbf16>) {
  ^bb0(%in: f32, %out_elem: bf16):
    %trunc = arith.truncf %in : f32 to bf16
    linalg.yield %trunc : bf16
  } -> tensor<32xbf16>
  
  return %truncf : tensor<32xbf16>
}

// CHECK-NOT: linalg.generic {{.*}} attributes {truncf_bf16}
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>)
// CHECK-SAME: outs(%arg2 : tensor<32xbf16>)
// CHECK: ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT_ELEM:.*]]: bf16):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN0]], %[[IN1]] : f32
// CHECK:   %[[TRUNC:.*]] = arith.truncf %[[MUL]] : f32 to bf16
// CHECK:   linalg.yield %[[TRUNC]] : bf16
// CHECK: return %[[RESULT]] : tensor<32xbf16>
