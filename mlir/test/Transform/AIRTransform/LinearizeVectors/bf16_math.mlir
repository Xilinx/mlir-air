//===- bf16_math.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// With bf16_math, an f32 product of f32 values takes its factors rounded to
// bf16, and at 32 lanes on AIE2P becomes aievec.mul_elem; tanh runs on its
// argument rounded to bf16. (A 32-lane f32 -> bf16 truncf is aievec.srs.)
// CHECK-LABEL: @f32_mul_tanh
// CHECK: %[[A:.*]] = aievec.srs %{{.*}}, %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: %[[B:.*]] = aievec.srs %{{.*}}, %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: %[[M:.*]] = aievec.mul_elem %[[A]], %[[B]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK: %[[T:.*]] = aievec.srs %[[M]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: %[[H:.*]] = math.tanh %[[T]] : vector<32xbf16>
// CHECK: arith.extf %[[H]] : vector<32xbf16> to vector<32xf32>
// CHECK-NOT: arith.mulf
func.func @f32_mul_tanh(%a: memref<32xf32>, %b: memref<32xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %bv = vector.transfer_read %b[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %m = arith.mulf %av, %bv : vector<32xf32>
  %t = math.tanh %m : vector<32xf32>
  vector.transfer_write %t, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 bf16_math : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// With contract as well, a multiply-add of f32 values becomes one
// mac_elem on the rounded factors.
// CHECK-LABEL: @f32_mul_add
// CHECK: aievec.mac_elem %{{.*}}, %{{.*}}, %{{.*}} : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
func.func @f32_mul_add(%a: memref<32xf32>, %b: memref<32xf32>, %c: memref<32xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %bv = vector.transfer_read %b[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %cv = vector.transfer_read %c[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %m = arith.mulf %av, %bv : vector<32xf32>
  %r = arith.addf %m, %cv : vector<32xf32>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 contract bf16_math : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Without bf16_math the f32 product and tanh are left in f32.
// CHECK-LABEL: @no_bf16_math
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK: math.tanh %{{.*}} : vector<32xf32>
// CHECK-NOT: aievec.mul_elem
func.func @no_bf16_math(%a: memref<32xf32>, %b: memref<32xf32>, %o: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %bv = vector.transfer_read %b[%c0], %p {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
  %m = arith.mulf %av, %bv : vector<32xf32>
  %t = math.tanh %m : vector<32xf32>
  vector.transfer_write %t, %o[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
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

// A bf16 value widened and narrowed back is the value itself.
// CHECK-LABEL: @trunc_of_ext
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<32xbf16>
func.func @trunc_of_ext(%a: memref<32xbf16>, %o: memref<32xbf16>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : bf16
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
  %e = arith.extf %av : vector<32xbf16> to vector<32xf32>
  %t = arith.truncf %e : vector<32xf32> to vector<32xbf16>
  %r = arith.addf %t, %t : vector<32xbf16>
  vector.transfer_write %r, %o[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16>
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

// Only an exact round trip folds: a rounding mode, or a narrower source type,
// keeps the truncf.
// CHECK-LABEL: @trunc_of_ext_kept
// CHECK: arith.truncf %{{.*}} toward_zero : vector<16xf32> to vector<16xbf16>
// CHECK: %[[E:.*]] = arith.extf %{{.*}} : vector<16xf16> to vector<16xf32>
// CHECK: arith.truncf %[[E]] : vector<16xf32> to vector<16xbf16>
func.func @trunc_of_ext_kept(%a: memref<16xbf16>, %h: memref<16xf16>, %o: memref<16xbf16>, %o2: memref<16xbf16>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : bf16
  %ph = arith.constant 0.0 : f16
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<16xbf16>, vector<16xbf16>
  %hv = vector.transfer_read %h[%c0], %ph {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
  %e = arith.extf %av : vector<16xbf16> to vector<16xf32>
  %t = arith.truncf %e toward_zero : vector<16xf32> to vector<16xbf16>
  %eh = arith.extf %hv : vector<16xf16> to vector<16xf32>
  %th = arith.truncf %eh : vector<16xf32> to vector<16xbf16>
  vector.transfer_write %t, %o[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16>
  vector.transfer_write %th, %o2[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16>
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

// On AIE2 the factors are still rounded, and the product stays an arith.mulf
// of widened bf16 values (16 lanes).
// CHECK-LABEL: @aie2_bf16_math
// CHECK: %[[A:.*]] = arith.truncf %{{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: %[[AW:.*]] = arith.extf %[[A]] : vector<16xbf16> to vector<16xf32>
// CHECK: %[[B:.*]] = arith.truncf %{{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: %[[BW:.*]] = arith.extf %[[B]] : vector<16xbf16> to vector<16xf32>
// CHECK: arith.mulf %[[AW]], %[[BW]] : vector<16xf32>
// CHECK-NOT: aievec
func.func @aie2_bf16_math(%a: memref<16xf32>, %b: memref<16xf32>, %o: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
  %bv = vector.transfer_read %b[%c0], %p {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
  %m = arith.mulf %av, %bv : vector<16xf32>
  vector.transfer_write %m, %o[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f bf16_math : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A GEMM epilogue h = bf16(0.5 * c * (1 + tanh(k1 * (c + k2 * c^3)))) on an
// 8x4 f32 accumulator tile: every product is a mul_elem or mac_elem, tanh runs
// on bf16, and the result leaves through srs.
// CHECK-LABEL: @gelu_epilogue
// CHECK-NOT: arith.mulf
// CHECK-DAG: aievec.mul_elem
// CHECK-DAG: aievec.mac_elem
// CHECK-NOT: aievec.srs %cst
// CHECK-DAG: math.tanh %{{.*}} : vector<32xbf16>
// A constant that holds bf16 values is used as a bf16 constant.
// CHECK-DAG: %[[HALF:.*]] = arith.constant dense<5.000000e-01> : vector<32xbf16>
// CHECK-DAG: aievec.mul_elem %{{.*}}, %[[HALF]]
// CHECK: %[[H:.*]] = aievec.srs %{{.*}}, %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: vector.transfer_write
func.func @gelu_epilogue(%acc: memref<8x4xf32>, %h: memref<8x4xbf16>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %k1 = arith.constant dense<0.797884583> : vector<8x4xf32>
  %k2 = arith.constant dense<4.471500e-02> : vector<8x4xf32>
  %half = arith.constant dense<5.000000e-01> : vector<8x4xf32>
  %one = arith.constant dense<1.000000e+00> : vector<8x4xf32>
  %c = vector.transfer_read %acc[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x4xf32>, vector<8x4xf32>
  %c2 = arith.mulf %c, %c : vector<8x4xf32>
  %c3 = arith.mulf %c2, %c : vector<8x4xf32>
  %kc3 = arith.mulf %k2, %c3 : vector<8x4xf32>
  %s = arith.addf %c, %kc3 : vector<8x4xf32>
  %z = arith.mulf %k1, %s : vector<8x4xf32>
  %t = math.tanh %z : vector<8x4xf32>
  %t1 = arith.addf %one, %t : vector<8x4xf32>
  %hc = arith.mulf %half, %c : vector<8x4xf32>
  %g = arith.mulf %hc, %t1 : vector<8x4xf32>
  %gb = arith.truncf %g : vector<8x4xf32> to vector<8x4xbf16>
  vector.transfer_write %gb, %h[%c0, %c0] {in_bounds = [true, true]} : vector<8x4xbf16>, memref<8x4xbf16>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.linearize_vectors %f arch = "aie2p" f32_lanes = 32 contract bf16_math : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
