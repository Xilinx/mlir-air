//===- transposed_broadcast_payload.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// A value per row, spread over its row: the broadcast and the transpose that
// puts the value along the row are one shuffle of the eight values.

// CHECK-LABEL: @per_row
// CHECK: %[[S:.*]] = vector.transfer_read {{.*}} : memref<8xf32>, vector<8xf32>
// CHECK: %[[R:.*]] = vector.shuffle %[[S]], %{{[^ ]*}} [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] : vector<8xf32>, vector<8xf32>
// CHECK: arith.mulf %{{.*}}, %[[R]] : vector<16xf32>
// CHECK: vector.shuffle %[[S]], %{{[^ ]*}} [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3] : vector<8xf32>, vector<8xf32>
func.func @per_row(%s: memref<8xf32>, %a: memref<64xf32>, %o: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %sv = vector.transfer_read %s[%c0], %p {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  %b = vector.broadcast %sv : vector<8xf32> to vector<8x8xf32>
  %t = vector.transpose %b, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
  %tf = vector.shape_cast %t : vector<8x8xf32> to vector<64xf32>
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %r = arith.mulf %av, %tf : vector<64xf32>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
  return
}

// A shuffle of 2-D vectors picks whole rows: its mask does not index lanes,
// so the lanes are not traced through it, and nothing is rewritten.

// CHECK-LABEL: @row_shuffle
// CHECK: vector.shuffle
// CHECK-SAME: [1, 0] : vector<2x4xf32>, vector<2x4xf32>
// CHECK-NEXT: vector.transpose
func.func @row_shuffle(%s: memref<8xf32>, %a: memref<8xf32>, %o: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %sv = vector.transfer_read %s[%c0], %p {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  %s2 = vector.shape_cast %sv : vector<8xf32> to vector<2x4xf32>
  %sh = vector.shuffle %s2, %s2 [1, 0] : vector<2x4xf32>, vector<2x4xf32>
  %t = vector.transpose %sh, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  %tf = vector.shape_cast %t : vector<4x2xf32> to vector<8xf32>
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  %r = arith.mulf %av, %tf : vector<8xf32>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  return
}
