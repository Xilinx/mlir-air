//===- air_transform_payload.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>

// A loop of at most max_trips trips with no contraction is unrolled; a longer
// one, or one with a contraction, is left.
// CHECK-LABEL: @unroll
// CHECK-COUNT-4: vector.transfer_write %{{.*}}, %arg0[%{{.*}}, %c0] {in_bounds = [true]} : vector<8xf32>
// CHECK: scf.for {{.*}} to %c8
// CHECK: scf.for {{.*}} to %c4
// CHECK: vector.contract
func.func @unroll(%o: memref<8x8xf32>, %p: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %z = arith.constant dense<0.0> : vector<8xf32>
  %f0 = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    vector.transfer_write %z, %o[%i, %c0] {in_bounds = [true]} : vector<8xf32>, memref<8x8xf32>
  }
  scf.for %i = %c0 to %c8 step %c1 {
    vector.transfer_write %z, %p[%i, %c0] {in_bounds = [true]} : vector<8xf32>, memref<8x8xf32>
  }
  %z2 = arith.constant dense<0.0> : vector<8x8xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %z2) -> (vector<8x8xf32>) {
    %x = vector.transfer_read %p[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
    %s = vector.contract {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %x, %x, %acc : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
    scf.yield %s : vector<8x8xf32>
  }
  vector.transfer_write %r, %o[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, memref<8x8xf32>
  return
}
