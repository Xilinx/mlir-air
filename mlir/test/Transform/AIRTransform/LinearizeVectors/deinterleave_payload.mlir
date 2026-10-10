//===- deinterleave_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// A read of one slot of interleaved pairs (the innermost dim fixed at k) is a
// read of the contiguous pairs and a shuffle taking every second element; through
// an expand_shape that only splits the innermost dim, the read goes to the
// unexpanded rows (the dead view is left for DCE).

// CHECK-LABEL: @slot1
// CHECK: %[[R:.*]] = vector.transfer_read %arg0[{{.*}} : memref<2x8x4x2xf32>, vector<8x4x2xf32>
// CHECK: %[[F:.*]] = vector.shape_cast %[[R]] : vector<8x4x2xf32> to vector<64xf32>
// CHECK: vector.shuffle %[[F]], %{{.*}} [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
func.func @slot1(%m: memref<2x8x4x2xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : f32
  %v = vector.transfer_read %m[%c0, %c0, %c0, %c1], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>} : memref<2x8x4x2xf32>, vector<8x4xf32>
  %f = vector.shape_cast %v : vector<8x4xf32> to vector<32xf32>
  vector.transfer_write %f, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

// CHECK-LABEL: @through_expand
// CHECK: vector.transfer_read %arg0[{{.*}} : memref<2x8x8xf32>, vector<8x8xf32>
// CHECK: vector.shuffle {{.*}} [0, 2, 4, 6
// CHECK-NOT: vector.transfer_read
func.func @through_expand(%m: memref<2x8x8xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %e = memref.expand_shape %m [[0], [1], [2, 3]] output_shape [2, 8, 4, 2] : memref<2x8x8xf32> into memref<2x8x4x2xf32>
  %v = vector.transfer_read %e[%c0, %c0, %c0, %c0], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>} : memref<2x8x4x2xf32>, vector<8x4xf32>
  %f = vector.shape_cast %v : vector<8x4xf32> to vector<32xf32>
  vector.transfer_write %f, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

// An f32 x f32 multiply-add stays as it is: the AIE multiply-accumulate takes
// widened bf16 operands only.
// CHECK-LABEL: @f32_mul_add
// CHECK-NOT: vector.fma
// CHECK: arith.mulf
// CHECK: arith.addf
func.func @f32_mul_add(%a: memref<16xf32>, %o: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
  %m = arith.mulf %v, %v : vector<16xf32>
  %s = arith.addf %m, %v : vector<16xf32>
  vector.transfer_write %s, %o[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  return
}
