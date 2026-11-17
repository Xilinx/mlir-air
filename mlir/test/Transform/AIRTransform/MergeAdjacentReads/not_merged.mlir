//===- not_merged.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// Reads that must stay apart; each function keeps its two reads.

// Elements of no bit width, and a memref of vectors.
// CHECK-LABEL: @index_elements
// CHECK-COUNT-2: vector.transfer_read
func.func @index_elements(%m: memref<2x4xindex>) -> (vector<1x4xindex>, vector<1x4xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0 : index
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x4xindex>, vector<1x4xindex>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x4xindex>, vector<1x4xindex>
  return %a, %b : vector<1x4xindex>, vector<1x4xindex>
}
// CHECK-LABEL: @vector_elements
// CHECK-COUNT-2: vector.transfer_read
func.func @vector_elements(%m: memref<2x4xvector<1xf32>>) -> (vector<1x1xf32>, vector<1x1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant dense<0.0> : vector<1xf32>
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true]} : memref<2x4xvector<1xf32>>, vector<1x1xf32>
  %b = vector.transfer_read %m[%c0, %c1], %p {in_bounds = [true]} : memref<2x4xvector<1xf32>>, vector<1x1xf32>
  return %a, %b : vector<1x1xf32>, vector<1x1xf32>
}

// A call between the reads may write anything; so may a write in a loop
// between them, or one through another view of the memory.
func.func private @g()
// CHECK-LABEL: @call_between
// CHECK-COUNT-2: vector.transfer_read
func.func @call_between(%m: memref<2x16xbf16, 2>) -> (vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  func.call @g() : () -> ()
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b : vector<1x16xbf16>, vector<1x16xbf16>
}
// CHECK-LABEL: @write_in_loop
// CHECK-COUNT-2: vector.transfer_read {{.*}} vector<1x16xbf16>
func.func @write_in_loop(%v: vector<1x16xbf16>, %n: index) -> (vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %m = memref.alloc() : memref<2x16xbf16, 2>
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  scf.for %i = %c0 to %n step %c1 {
    vector.transfer_write %v, %m[%c1, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<2x16xbf16, 2>
  }
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b : vector<1x16xbf16>, vector<1x16xbf16>
}
// CHECK-LABEL: @write_through_view
// CHECK-COUNT-2: vector.transfer_read {{.*}} vector<1x16xbf16>
func.func @write_through_view(%v: vector<16xbf16>) -> (vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %m = memref.alloc() : memref<2x16xbf16, 2>
  %row = memref.subview %m[1, 0] [1, 16] [1, 1] : memref<2x16xbf16, 2> to memref<16xbf16, strided<[1], offset: 16>, 2>
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  vector.transfer_write %v, %row[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: 16>, 2>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b : vector<1x16xbf16>, vector<1x16xbf16>
}

// Rows 32 apart are not one piece; a masked, a maybe out-of-bounds, and a
// differently padded read are left alone.
// CHECK-LABEL: @not_one_piece
// CHECK-COUNT-2: vector.transfer_read
func.func @not_one_piece(%m: memref<2x16xbf16, strided<[32, 1]>, 2>) -> (vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, strided<[32, 1]>, 2>, vector<1x16xbf16>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, strided<[32, 1]>, 2>, vector<1x16xbf16>
  return %a, %b : vector<1x16xbf16>, vector<1x16xbf16>
}
// CHECK-LABEL: @masked_or_padded
// CHECK-COUNT-6: vector.transfer_read
func.func @masked_or_padded(%m: memref<2x16xbf16, 2>, %k: vector<1x16xi1>) -> (vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %q = arith.constant 1.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p, %k {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %b = vector.transfer_read %m[%c1, %c0], %p, %k {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %c = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, false]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %d = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, false]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %e = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %f = vector.transfer_read %m[%c1, %c0], %q {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b, %c, %d, %e, %f : vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>
}
