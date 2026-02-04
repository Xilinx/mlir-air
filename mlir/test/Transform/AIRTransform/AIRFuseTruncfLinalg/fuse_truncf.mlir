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
// Producer's init (tensor.empty) is type-converted to bf16
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xbf16>
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<16xbf16>)
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
// Producer's init (tensor.empty for matmul) is type-converted to bf16
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x8xbf16>
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x8xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<8x8xbf16>)
// CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %[[ACC:.*]]: bf16):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:   arith.extf %[[ACC]] : bf16 to f32
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
// Producer's init (tensor.empty) is type-converted to bf16
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xbf16>
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<32xbf16>)
// CHECK: ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT_ELEM:.*]]: bf16):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN0]], %[[IN1]] : f32
// CHECK:   %[[TRUNC:.*]] = arith.truncf %[[MUL]] : f32 to bf16
// CHECK:   linalg.yield %[[TRUNC]] : bf16
// CHECK: return %[[RESULT]] : tensor<32xbf16>

// Test 4: Fuse truncf into mixed-precision matmul (bf16 inputs, f32 accumulator)
// This tests the pattern from Triton matmul where inputs are bf16 but accumulation is f32
// CHECK-LABEL: @fuse_truncf_into_mixed_precision_matmul
func.func @fuse_truncf_into_mixed_precision_matmul(%a: tensor<8x16xbf16>, %b: tensor<16x8xbf16>, %c: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  
  // Mixed-precision matmul: bf16 inputs, f32 accumulator
  %matmul = linalg.matmul
    ins(%a, %b : tensor<8x16xbf16>, tensor<16x8xbf16>)
    outs(%fill : tensor<8x8xf32>) -> tensor<8x8xf32>
  
  // Truncf operation: truncates f32 result to bf16
  %truncf = linalg.generic {truncf_mixed_matmul,
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

// CHECK-NOT: linalg.generic {{.*}} attributes {truncf_mixed_matmul}
// Producer's init (linalg.fill with 0.0) is type-converted to bf16 fill
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x8xbf16>
// CHECK: %[[TRUNC_CST:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[TRUNC_CST]] : bf16) outs(%[[EMPTY]] : tensor<8x8xbf16>)
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xbf16>, tensor<16x8xbf16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<8x8xbf16>)
// CHECK: ^bb0(%[[LHS:.*]]: bf16, %[[RHS:.*]]: bf16, %[[ACC:.*]]: bf16):
// CHECK:   %[[TRUNC:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK:   linalg.yield %[[TRUNC]] : bf16
// CHECK: return %[[RESULT]] : tensor<8x8xbf16>

// Test 5: Fuse truncf with 
// CHECK-LABEL: @fuse_truncf_bufferized
func.func @fuse_truncf_bufferized(%a: tensor<256x256xbf16>, %b: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  // After bufferize_to_allocation, fill's output comes from bufferization.to_tensor
  %alloc = memref.alloc() : memref<256x256xf32, 1>
  %tensor_from_alloc = bufferization.to_tensor %alloc restrict writable : memref<256x256xf32, 1> to tensor<256x256xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%tensor_from_alloc : tensor<256x256xf32>) -> tensor<256x256xf32>

  // Mixed-precision matmul: bf16 inputs, f32 accumulator
  %matmul = linalg.matmul
    ins(%a, %b : tensor<256x256xbf16>, tensor<256x256xbf16>)
    outs(%fill : tensor<256x256xf32>) -> tensor<256x256xf32>

  // Truncf operation with tensor.empty() as output
  %empty_bf16 = tensor.empty() : tensor<256x256xbf16>
  %truncf = linalg.generic {truncf_bufferized,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul : tensor<256x256xf32>)
    outs(%empty_bf16 : tensor<256x256xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %trunc = arith.truncf %in : f32 to bf16
    linalg.yield %trunc : bf16
  } -> tensor<256x256xbf16>

  memref.dealloc %alloc : memref<256x256xf32, 1>
  return %truncf : tensor<256x256xbf16>
}

// CHECK-NOT: linalg.generic {{.*}} attributes {truncf_bufferized}
// Producer's init (linalg.fill from bufferized alloc) is type-converted to bf16 fill
// CHECK: tensor.empty() : tensor<256x256xbf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<256x256xbf16>
// CHECK: %[[TRUNC_CST:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[TRUNC_CST]] : bf16) outs(%[[EMPTY2]] : tensor<256x256xbf16>)
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<256x256xbf16>, tensor<256x256xbf16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<256x256xbf16>)
// CHECK: ^bb0(%[[LHS:.*]]: bf16, %[[RHS:.*]]: bf16, %[[ACC:.*]]: bf16):
// CHECK:   %[[TRUNC:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK:   linalg.yield %[[TRUNC]] : bf16
// CHECK: return %[[RESULT]] : tensor<256x256xbf16>
