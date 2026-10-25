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
// A matmul producer gives a linalg.matmul with a bf16 result.
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x8xbf16>
// CHECK: %[[RESULT:.*]] = linalg.matmul
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x8xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<8x8xbf16>)
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
// CHECK: %[[RESULT:.*]] = linalg.matmul
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xbf16>, tensor<16x8xbf16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<8x8xbf16>)
// CHECK: return %[[RESULT]] : tensor<8x8xbf16>

// Test 5: Fuse truncf into a bufferized matmul with bf16 inputs and f32 accumulator
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
// CHECK: %[[RESULT:.*]] = linalg.matmul
// CHECK-SAME: ins(%arg0, %arg1 : tensor<256x256xbf16>, tensor<256x256xbf16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<256x256xbf16>)
// CHECK: return %[[RESULT]] : tensor<256x256xbf16>

// Test 6: An elementwise producer on 2-D tensors stays a generic: the fused
// op multiplies and adds elementwise and truncates, with no reduction.
// CHECK-LABEL: @fuse_truncf_into_2d_elementwise
// CHECK-NOT: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: %[[M:.*]] = arith.mulf
// CHECK: %[[A:.*]] = arith.addf %[[M]]
// CHECK: %[[T:.*]] = arith.truncf %[[A]] : f32 to bf16
// CHECK: linalg.yield %[[T]] : bf16
// CHECK-NOT: linalg.matmul
// CHECK: return
func.func @fuse_truncf_into_2d_elementwise(%a: tensor<64x512xf32>, %b: tensor<64x512xf32>, %c: tensor<64x512xf32>, %out: tensor<64x512xbf16>) -> tensor<64x512xbf16> {
  %empty = tensor.empty() : tensor<64x512xf32>
  %fma = linalg.generic {producer_2d,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b, %c : tensor<64x512xf32>, tensor<64x512xf32>, tensor<64x512xf32>)
    outs(%empty : tensor<64x512xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %m, %z : f32
    linalg.yield %s : f32
  } -> tensor<64x512xf32>
  %truncf = linalg.generic {truncf_2d,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%fma : tensor<64x512xf32>) outs(%out : tensor<64x512xbf16>) {
  ^bb0(%in: f32, %o: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<64x512xbf16>
  return %truncf : tensor<64x512xbf16>
}

// Test 7: Fuse truncf into a contraction that reads B as [n, k]. Its maps are
// not a matmul's, so the fused op stays a generic with those maps.
// CHECK-LABEL: @fuse_truncf_into_transposed_b
// CHECK-NOT: linalg.matmul
// CHECK: linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%arg0, %arg1 : tensor<8x16xbf16>, tensor<8x16xbf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<8x8xbf16>)
// CHECK-NOT: linalg.matmul
func.func @fuse_truncf_into_transposed_b(%a: tensor<8x16xbf16>, %b: tensor<8x16xbf16>, %c: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %mm = linalg.generic {producer_tb,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b : tensor<8x16xbf16>, tensor<8x16xbf16>) outs(%fill : tensor<8x8xf32>) {
  ^bb0(%x: bf16, %y: bf16, %acc: f32):
    %xe = arith.extf %x : bf16 to f32
    %ye = arith.extf %y : bf16 to f32
    %m = arith.mulf %xe, %ye : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  %truncf = linalg.generic {truncf_tb,
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm : tensor<8x8xf32>) outs(%c : tensor<8x8xbf16>) {
  ^bb0(%in: f32, %out_elem: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<8x8xbf16>
  return %truncf : tensor<8x8xbf16>
}
