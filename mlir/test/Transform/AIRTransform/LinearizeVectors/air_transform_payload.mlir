//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// An int8 tile scaled and offset per column, with the scale and offset
// broadcast along the rows: w = (bf16(0x4300 | q) - 128) * s + m.
// Everything elementwise becomes rank 1; the scale and offset reads lose their
// broadcast permutation map and are replicated by a shuffle.

// CHECK-LABEL: @dequant_tile
// CHECK-DAG: %[[Q:.*]] = vector.transfer_read %arg0{{.*}} : memref<1x4x8xi8>, vector<1x4x8xi8>
// CHECK-DAG: %[[S:.*]] = vector.transfer_read %arg1{{.*}} {in_bounds = [true, true, true]} : memref<1x1x8xbf16>, vector<1x1x8xbf16>
// CHECK-DAG: %[[Q1:.*]] = vector.shape_cast %[[Q]] : vector<1x4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[S1:.*]] = vector.shape_cast %[[S]] : vector<1x1x8xbf16> to vector<8xbf16>
// CHECK-DAG: %[[SB:.*]] = vector.shuffle %[[S1]], %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
// CHECK: %[[EXT:.*]] = arith.extsi %[[Q1]] : vector<32xi8> to vector<32xi16>
// CHECK: %[[OR:.*]] = arith.ori %[[EXT]], %{{.*}} : vector<32xi16>
// CHECK: %[[QB:.*]] = arith.bitcast %[[OR]] : vector<32xi16> to vector<32xbf16>
// CHECK: %[[SUB:.*]] = arith.subf %[[QB]], %{{.*}} : vector<32xbf16>
// CHECK: %[[MUL:.*]] = arith.mulf %[[SUB]], %[[SB]] : vector<32xbf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[MUL]], %{{.*}} : vector<32xbf16>
// CHECK: %[[W:.*]] = vector.shape_cast %[[ADD]] : vector<32xbf16> to vector<1x4x8xbf16>
// CHECK: vector.transfer_write %[[W]], %arg3
func.func @dequant_tile(%q: memref<1x4x8xi8>, %s: memref<1x1x8xbf16>,
                        %m: memref<1x1x8xbf16>, %w: memref<1x4x8xbf16>) {
  %c0 = arith.constant 0 : index
  %pi8 = arith.constant 0 : i8
  %pbf = arith.constant 0.0 : bf16
  %magic = arith.constant dense<17152> : vector<1x4x8xi16>
  %c128 = arith.constant dense<128.0> : vector<1x4x8xbf16>
  %qv = vector.transfer_read %q[%c0, %c0, %c0], %pi8 {in_bounds = [true, true, true]} : memref<1x4x8xi8>, vector<1x4x8xi8>
  %sv = vector.transfer_read %s[%c0, %c0, %c0], %pbf {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, 0, d2)>} : memref<1x1x8xbf16>, vector<1x4x8xbf16>
  %mv = vector.transfer_read %m[%c0, %c0, %c0], %pbf {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, 0, d2)>} : memref<1x1x8xbf16>, vector<1x4x8xbf16>
  %e = arith.extsi %qv : vector<1x4x8xi8> to vector<1x4x8xi16>
  %o = arith.ori %e, %magic : vector<1x4x8xi16>
  %b = arith.bitcast %o : vector<1x4x8xi16> to vector<1x4x8xbf16>
  %d = arith.subf %b, %c128 : vector<1x4x8xbf16>
  %ms = arith.mulf %d, %sv : vector<1x4x8xbf16>
  %r = arith.addf %ms, %mv : vector<1x4x8xbf16>
  vector.transfer_write %r, %w[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x8xbf16>, memref<1x4x8xbf16>
  return
}

// A read that is not a broadcast, and a contraction, keep their n-D types.
// CHECK-LABEL: @untouched
// CHECK: vector.transfer_read {{.*}} : memref<4x8xf32>, vector<4x8xf32>
// CHECK: vector.contract
// CHECK-SAME: vector<4x8xf32>
func.func @untouched(%a: memref<4x8xf32>, %b: memref<8x4xf32>, %c: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
  %bv = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x4xf32>, vector<8x4xf32>
  %cv = vector.transfer_read %c[%c0, %c0], %p {in_bounds = [true, true]} : memref<4x4xf32>, vector<4x4xf32>
  %r = vector.contract {indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %av, %bv, %cv : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  vector.transfer_write %r, %c[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
  return
}

// A 4x8 block of rows 64 wide is not contiguous: it is read row by row and
// assembled with shuffles, never as 32 consecutive elements.
// CHECK-LABEL: @non_contiguous_block
// CHECK-NOT: vector<32xi16>, memref
// CHECK-COUNT-4: vector.transfer_read %arg0{{.*}} : memref<4x64xi16>, vector<8xi16>
// CHECK: arith.ori {{.*}} : vector<32xi16>
func.func @non_contiguous_block(%a: memref<4x64xi16>, %b: memref<32xi16>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i16
  %magic = arith.constant dense<17152> : vector<4x8xi16>
  %v = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<4x64xi16>, vector<4x8xi16>
  %o = arith.ori %v, %magic : vector<4x8xi16>
  %f = vector.shape_cast %o : vector<4x8xi16> to vector<32xi16>
  vector.transfer_write %f, %b[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16>
  return
}

// A 4-bit unpack spelled with shifts (each 32-bit word replicated over 8
// lanes, shifted right by 0, 4, ..., 28 and masked) becomes an i4 bitcast
// and extui.
// CHECK-LABEL: @nibble_unpack
// CHECK-NOT: arith.shrsi
// CHECK: %[[B:.*]] = vector.bitcast %{{.*}} : vector<4xi32> to vector<16xi8>
// CHECK: %[[P:.*]] = vector.shuffle %[[B]]
// CHECK: %[[N:.*]] = vector.bitcast %[[P]] : vector<32xi8> to vector<64xi4>
// CHECK: arith.extui %[[N]] : vector<64xi4> to vector<64xi8>
func.func @nibble_unpack(%w: memref<4x1xi32>, %o: memref<4x8xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %sh = arith.constant dense<[[0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28]]> : vector<4x8xi32>
  %c15 = arith.constant dense<15> : vector<4x8xi32>
  %v = vector.transfer_read %w[%c0, %c0], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, 0)>} : memref<4x1xi32>, vector<4x8xi32>
  %x = arith.shrsi %v, %sh : vector<4x8xi32>
  %m = arith.andi %x, %c15 : vector<4x8xi32>
  vector.transfer_write %m, %o[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xi32>, memref<4x8xi32>
  return
}

// A column of a buffer (stride 8) is read element by element.
// CHECK-LABEL: @strided_column
// CHECK-COUNT-4: memref.load %arg0
// CHECK: vector.from_elements
func.func @strided_column(%w: memref<4xi32, strided<[8]>>, %o: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %v = vector.transfer_read %w[%c0], %p {in_bounds = [true]} : memref<4xi32, strided<[8]>>, vector<4xi32>
  vector.transfer_write %v, %o[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
  return
}

// A contiguous read whose map skips trailing unit dims (left by folding a
// subview into it) gets a minor identity map again.
// CHECK-LABEL: @trailing_unit_dims
// CHECK: %[[C:.*]] = memref.collapse_shape %arg0 {{\[\[}}0], [1, 2]]
// CHECK: vector.transfer_read %[[C]]{{.*}} : memref<8x8xi32>, vector<4xi32>
func.func @trailing_unit_dims(%w: memref<8x8x1xi32>, %o: memref<4xi32>, %i: index) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %v = vector.transfer_read %w[%i, %c0, %c0], %p {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : memref<8x8x1xi32>, vector<4xi32>
  vector.transfer_write %v, %o[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
  return
}

// A strided read not known to be in bounds keeps its padding semantics, so it
// is not unrolled into memref.load.
// CHECK-LABEL: @strided_column_not_in_bounds
// CHECK-NOT: vector.from_elements
// CHECK: vector.transfer_read
func.func @strided_column_not_in_bounds(%w: memref<4xi32, strided<[8]>>, %o: memref<4xi32>, %i: index) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %v = vector.transfer_read %w[%i], %p {in_bounds = [false]} : memref<4xi32, strided<[8]>>, vector<4xi32>
  vector.transfer_write %v, %o[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
  return
}

// Trailing unit dims indexed by something other than the constant 0 are not
// collapsed away.
// CHECK-LABEL: @trailing_unit_dims_nonzero_index
// CHECK-NOT: memref.collapse_shape
func.func @trailing_unit_dims_nonzero_index(%w: memref<8x8x1xi32>, %o: memref<4xi32>, %i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %v = vector.transfer_read %w[%i, %c0, %j], %p {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : memref<8x8x1xi32>, vector<4xi32>
  vector.transfer_write %v, %o[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
  return
}

// The same spelling on i8 words is left as shifts: its result is already i8.
// CHECK-LABEL: @nibble_unpack_i8
// CHECK-NOT: vector<{{[0-9]+}}xi4>
// CHECK: arith.shrsi
func.func @nibble_unpack_i8(%w: memref<16x1xi8>, %o: memref<16x2xi8>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i8
  %sh = arith.constant dense<[[0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4], [0, 4]]> : vector<16x2xi8>
  %c15 = arith.constant dense<15> : vector<16x2xi8>
  %v = vector.transfer_read %w[%c0, %c0], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, 0)>} : memref<16x1xi8>, vector<16x2xi8>
  %x = arith.shrsi %v, %sh : vector<16x2xi8>
  %m = arith.andi %x, %c15 : vector<16x2xi8>
  vector.transfer_write %m, %o[%c0, %c0] {in_bounds = [true, true]} : vector<16x2xi8>, memref<16x2xi8>
  return
}
