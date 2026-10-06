//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// 64 4-bit fields, one unpack's worth. For AIE2 the unpack is spelled as an
// i4 bitcast and extui.
// CHECK-LABEL: @nibble_unpack_64
// CHECK-NOT: aievec.unpack
// CHECK: %[[N:.*]] = vector.bitcast %{{.*}} : vector<8xi32> to vector<64xi4>
// CHECK: arith.extui %[[N]] : vector<64xi4> to vector<64xi8>
func.func @nibble_unpack_64(%w: memref<8x1xi32>, %o: memref<8x8xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %sh = arith.constant dense<[[0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28]]> : vector<8x8xi32>
  %c15 = arith.constant dense<15> : vector<8x8xi32>
  %v = vector.transfer_read %w[%c0, %c0], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, 0)>} : memref<8x1xi32>, vector<8x8xi32>
  %x = arith.shrsi %v, %sh : vector<8x8xi32>
  %m = arith.andi %x, %c15 : vector<8x8xi32>
  vector.transfer_write %m, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xi32>, memref<8x8xi32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// For AIE2P it is aievec.unpack.
// CHECK-LABEL: @nibble_unpack_64
// CHECK: %[[B:.*]] = vector.bitcast %{{.*}} : vector<8xi32> to vector<32xi8>
// CHECK: %[[U:.*]] = aievec.unpack %[[B]] : vector<32xi8>, vector<64xi8>
// CHECK: arith.extsi %[[U]] : vector<64xi8> to vector<64xi32>
func.func @nibble_unpack_64(%w: memref<8x1xi32>, %o: memref<8x8xi32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %sh = arith.constant dense<[[0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28], [0, 4, 8, 12, 16, 20, 24, 28]]> : vector<8x8xi32>
  %c15 = arith.constant dense<15> : vector<8x8xi32>
  %v = vector.transfer_read %w[%c0, %c0], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, 0)>} : memref<8x1xi32>, vector<8x8xi32>
  %x = arith.shrsi %v, %sh : vector<8x8xi32>
  %m = arith.andi %x, %c15 : vector<8x8xi32>
  vector.transfer_write %m, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xi32>, memref<8x8xi32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Rows that may be out of bounds are not split: each row read would keep only
// the innermost dim's bound.
// CHECK-LABEL: @split_needs_in_bounds_rows
// CHECK: vector.transfer_read %{{.*}} {in_bounds = [false, true]} : memref<?x64xi16>, vector<4x8xi16>
// CHECK-NOT: vector.insert
func.func @split_needs_in_bounds_rows(%a: memref<?x64xi16>, %b: memref<32xi16>, %i: index) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i16
  %v = vector.transfer_read %a[%i, %c0], %p {in_bounds = [false, true]} : memref<?x64xi16>, vector<4x8xi16>
  %s = vector.shape_cast %v : vector<4x8xi16> to vector<32xi16>
  vector.transfer_write %s, %b[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The same read with in-bounds rows is split, one read per row.
// CHECK-LABEL: @split_in_bounds_rows
// CHECK-COUNT-4: vector.transfer_read {{.*}} : memref<?x64xi16>, vector<8xi16>
func.func @split_in_bounds_rows(%a: memref<?x64xi16>, %b: memref<32xi16>, %i: index) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i16
  %v = vector.transfer_read %a[%i, %c0], %p {in_bounds = [true, false]} : memref<?x64xi16>, vector<4x8xi16>
  %s = vector.shape_cast %v : vector<4x8xi16> to vector<32xi16>
  vector.transfer_write %s, %b[%c0] {in_bounds = [true]} : vector<32xi16>, memref<32xi16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A truncf with a rounding mode other than the default is not an srs.
// CHECK-LABEL: @truncf_rounding_mode
// CHECK: aievec.srs %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: arith.truncf %{{.*}} toward_zero : vector<32xf32> to vector<32xbf16>
func.func @truncf_rounding_mode(%a: memref<32xf32>, %b: memref<32xbf16>, %c: memref<32xbf16>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %t = arith.truncf %v : vector<32xf32> to vector<32xbf16>
  %z = arith.truncf %v toward_zero : vector<32xf32> to vector<32xbf16>
  vector.transfer_write %t, %b[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16>
  vector.transfer_write %z, %c[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The same at 64 lanes.
// CHECK-LABEL: @truncf_rounding_mode_64
// CHECK-COUNT-2: aievec.srs %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: arith.truncf %{{.*}} toward_zero : vector<64xf32> to vector<64xbf16>
func.func @truncf_rounding_mode_64(%a: memref<64xf32>, %b: memref<64xbf16>, %c: memref<64xbf16>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %t = arith.truncf %v : vector<64xf32> to vector<64xbf16>
  %z = arith.truncf %v toward_zero : vector<64xf32> to vector<64xbf16>
  vector.transfer_write %t, %b[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  vector.transfer_write %z, %c[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 64 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// At 32 lanes, without `contract` and without fast-math flags, a widened
// bf16 multiply and add stay separate: a mul_elem and an add.
// CHECK-LABEL: @mul_add_no_contract
// CHECK-NOT: aievec.mac_elem
// CHECK: aievec.mul_elem
// CHECK: arith.addf
func.func @mul_add_no_contract(%a: memref<32xbf16>, %b: memref<32xbf16>, %c: memref<32xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %pb = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %pb {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %bv = vector.transfer_read %b[%c0], %pb {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %cv = vector.transfer_read %c[%c0], %pf {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %ae = arith.extf %av : vector<32xbf16> to vector<32xf32>
  %be = arith.extf %bv : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ae, %be : vector<32xf32>
  %r = arith.addf %m, %cv : vector<32xf32>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// With the contract flag on both ops they become one 32-lane mac_elem.
// CHECK-LABEL: @mul_add_contract_flags
// CHECK: aievec.mac_elem %{{.*}}, %{{.*}}, %{{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK-NOT: arith.mulf
func.func @mul_add_contract_flags(%a: memref<32xbf16>, %b: memref<32xbf16>, %c: memref<32xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %pb = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %pb {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %bv = vector.transfer_read %b[%c0], %pb {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %cv = vector.transfer_read %c[%c0], %pf {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %ae = arith.extf %av : vector<32xbf16> to vector<32xf32>
  %be = arith.extf %bv : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ae, %be fastmath<contract> : vector<32xf32>
  %r = arith.addf %m, %cv fastmath<contract> : vector<32xf32>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// With the flag on the multiply only, they stay separate.
// CHECK-LABEL: @mul_add_mixed_flags
// CHECK-NOT: aievec.mac_elem
// CHECK: aievec.mul_elem
// CHECK: arith.addf
func.func @mul_add_mixed_flags(%a: memref<32xbf16>, %b: memref<32xbf16>, %c: memref<32xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %pb = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %pb {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %bv = vector.transfer_read %b[%c0], %pb {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %cv = vector.transfer_read %c[%c0], %pf {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %ae = arith.extf %av : vector<32xbf16> to vector<32xf32>
  %be = arith.extf %bv : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ae, %be fastmath<contract> : vector<32xf32>
  %r = arith.addf %m, %cv : vector<32xf32>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{f32_lanes = 32 needs arch = "aie2p"}}
    %r = transform.air.linearize_vectors %f f32_lanes = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{f32_lanes must be 16, 32 or 64}}
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 48 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{arch must be "aie2" or "aie2p"}}
    %r = transform.air.linearize_vectors %f arch = "aie3" : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
