//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>

// A loop's sum from zero, added to y after it: y, made before the loop,
// starts the sum.
// CHECK-LABEL: @loop
// CHECK: %[[O:.*]] = vector.transfer_read %arg0
// CHECK: %[[Y:.*]] = arith.mulf %[[O]], %arg3
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[Y]])
// CHECK: vector.contract {{.*}} %[[ACC]]
// CHECK: }
// CHECK-NOT: arith.addf
// CHECK: vector.transfer_write %[[R]], %arg0
func.func @loop(%o: memref<8x8xf32>, %p: memref<4x8x8xf32>, %v: memref<4x8x8xf32>, %c: vector<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %z = arith.constant dense<0.0> : vector<8x8xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %z) -> (vector<8x8xf32>) {
    %pa = vector.transfer_read %p[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x8x8xf32>, vector<8x8xf32>
    %va = vector.transfer_read %v[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x8x8xf32>, vector<8x8xf32>
    %s = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %pa, %va, %acc : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
    scf.yield %s : vector<8x8xf32>
  }
  %oa = vector.transfer_read %o[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  %y = arith.mulf %oa, %c : vector<8x8xf32>
  %out = arith.addf %y, %r : vector<8x8xf32>
  vector.transfer_write %out, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
  return
}

// The loop writes what y reads: y is left after the loop.
// CHECK-LABEL: @loop_writes_y
// CHECK: %[[Z:.*]] = arith.constant dense<0.0{{.*}}> : vector<8x8xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[Z]])
// CHECK: arith.addf
func.func @loop_writes_y(%o: memref<8x8xf32>, %p: memref<4x8x8xf32>, %v: memref<4x8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %z = arith.constant dense<0.0> : vector<8x8xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %z) -> (vector<8x8xf32>) {
    %pa = vector.transfer_read %p[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x8x8xf32>, vector<8x8xf32>
    %va = vector.transfer_read %v[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x8x8xf32>, vector<8x8xf32>
    vector.transfer_write %pa, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
    %s = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %pa, %va, %acc : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
    scf.yield %s : vector<8x8xf32>
  }
  %oa = vector.transfer_read %o[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  %out = arith.addf %oa, %r : vector<8x8xf32>
  vector.transfer_write %out, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
  return
}

// A chain of contractions from zero (an unrolled loop), added to y: y starts
// the chain, made before its first link; a write to another tile of the same
// memref in between does not stop it.
// CHECK-LABEL: @chain
// CHECK: %[[O:.*]] = vector.transfer_read %arg0[%c1
// CHECK: %[[Y:.*]] = arith.mulf %[[O]], %arg3
// CHECK: %[[S0:.*]] = vector.contract {{.*}} %[[Y]]
// CHECK: %[[S1:.*]] = vector.contract {{.*}} %[[S0]]
// CHECK-NOT: arith.addf
// CHECK: vector.transfer_write %[[S1]], %arg0[%c1
func.func @chain(%o: memref<2x8x8xf32>, %p: memref<2x8x8xf32>, %v: memref<2x8x8xf32>, %c: vector<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %z = arith.constant dense<0.0> : vector<8x8xf32>
  %p0 = vector.transfer_read %p[%c0, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<2x8x8xf32>, vector<8x8xf32>
  %v0 = vector.transfer_read %v[%c0, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<2x8x8xf32>, vector<8x8xf32>
  %s0 = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %p0, %v0, %z : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  vector.transfer_write %p0, %o[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<2x8x8xf32>
  %p1 = vector.transfer_read %p[%c1, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<2x8x8xf32>, vector<8x8xf32>
  %v1 = vector.transfer_read %v[%c1, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<2x8x8xf32>, vector<8x8xf32>
  %s1 = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %p1, %v1, %s0 : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  %oa = vector.transfer_read %o[%c1, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<2x8x8xf32>, vector<8x8xf32>
  %y = arith.mulf %oa, %c : vector<8x8xf32>
  %out = arith.addf %y, %s1 : vector<8x8xf32>
  vector.transfer_write %out, %o[%c1, %c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<2x8x8xf32>
  return
}

// Between the loop and y's read, a write of what y reads: y stays after it.
// CHECK-LABEL: @write_before_y
// CHECK: scf.for
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read %arg0
// CHECK: arith.addf
func.func @write_before_y(%o: memref<8x8xf32>, %p: memref<4x8x8xf32>, %v: memref<4x8x8xf32>, %w: vector<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %z = arith.constant dense<0.0> : vector<8x8xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %z) -> (vector<8x8xf32>) {
    %pa = vector.transfer_read %p[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x8x8xf32>, vector<8x8xf32>
    %va = vector.transfer_read %v[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x8x8xf32>, vector<8x8xf32>
    %s = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %pa, %va, %acc : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
    scf.yield %s : vector<8x8xf32>
  }
  vector.transfer_write %w, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
  %oa = vector.transfer_read %o[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  %out = arith.addf %oa, %r : vector<8x8xf32>
  vector.transfer_write %out, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
  return
}
