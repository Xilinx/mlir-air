//===- no_recompute_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/no_recompute.mlir' %s | FileCheck %s
// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s --check-prefix=PLAIN

// With no_recompute a producer is fused only into its only user, read through
// a permutation. Without it, all of these fuse.

#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

// The only user reads it element for element: fused either way.
// CHECK-LABEL: @permutation
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: linalg.generic
// PLAIN-LABEL: @permutation
// PLAIN: linalg.generic
// PLAIN-NOT: linalg.generic
func.func @permutation(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %m = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %0 = arith.mulf %x, %x : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  %r = linalg.generic {indexing_maps = [#id1, #id1, #id1], iterator_types = ["parallel"]}
      ins(%m, %b : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// Read through a broadcast: fusing would compute each row value once per
// column.
// CHECK-LABEL: @broadcast
// CHECK: %[[M:.*]] = linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %[[M]] :
// CHECK-NOT: arith.mulf
// CHECK: arith.subf
// PLAIN-LABEL: @broadcast
// PLAIN: linalg.generic
// PLAIN: arith.mulf
// PLAIN: arith.subf
// PLAIN-NOT: linalg.generic
func.func @broadcast(%a: tensor<4xf32>, %x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %e1 = tensor.empty() : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4x8xf32>
  %m = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%a : tensor<4xf32>) outs(%e1 : tensor<4xf32>) {
  ^bb0(%v: f32, %o: f32):
    %0 = arith.mulf %v, %v : f32
    linalg.yield %0 : f32
  } -> tensor<4xf32>
  %s = linalg.generic {indexing_maps = [#id2, #row, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<4x8xf32>, tensor<4xf32>) outs(%e2 : tensor<4x8xf32>) {
  ^bb0(%v: f32, %w: f32, %o: f32):
    %0 = arith.subf %v, %w : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %s : tensor<4x8xf32>
}

// A producer with another user stays for it.
// CHECK-LABEL: @multi_use
// CHECK: %[[M:.*]] = linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[M]], %{{.*}} :
// CHECK-NOT: arith.mulf
// CHECK: arith.addf
// CHECK: return %{{.*}}, %[[M]]
// PLAIN-LABEL: @multi_use
// PLAIN: linalg.generic
// PLAIN: arith.mulf
// PLAIN: linalg.generic
// PLAIN: arith.mulf
// PLAIN: arith.addf
func.func @multi_use(%a: tensor<8xf32>, %b: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %e = tensor.empty() : tensor<8xf32>
  %m = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %0 = arith.mulf %x, %x : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  %r = linalg.generic {indexing_maps = [#id1, #id1, #id1], iterator_types = ["parallel"]}
      ins(%m, %b : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  return %r, %m : tensor<8xf32>, tensor<8xf32>
}

// The only user reads it transposed, still each element once: fused.
// CHECK-LABEL: @transposed
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: linalg.generic
// PLAIN-LABEL: @transposed
// PLAIN: linalg.generic
// PLAIN-NOT: linalg.generic
func.func @transposed(%a: tensor<8x4xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %e1 = tensor.empty() : tensor<8x4xf32>
  %e2 = tensor.empty() : tensor<4x8xf32>
  %m = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<8x4xf32>) outs(%e1 : tensor<8x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %0 = arith.mulf %x, %x : f32
    linalg.yield %0 : f32
  } -> tensor<8x4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%m, %b : tensor<8x4xf32>, tensor<4x8xf32>) outs(%e2 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// The only user is inside a loop the producer is outside of: fused, the
// producer would be computed again every trip.
// CHECK-LABEL: @outside_loop
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: scf.for
// CHECK: linalg.generic
// CHECK-NOT: arith.mulf
// CHECK: arith.addf
// PLAIN-LABEL: @outside_loop
// PLAIN: scf.for
// PLAIN: linalg.generic
// PLAIN: arith.mulf
// PLAIN: arith.addf
func.func @outside_loop(%a: tensor<8xf32>, %b: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %e = tensor.empty() : tensor<8xf32>
  %m = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %0 = arith.mulf %x, %x : f32
    linalg.yield %0 : f32
  } -> tensor<8xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %b) -> (tensor<8xf32>) {
    %s = linalg.generic {indexing_maps = [#id1, #id1, #id1], iterator_types = ["parallel"]}
        ins(%m, %acc : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %0 = arith.addf %x, %y : f32
      linalg.yield %0 : f32
    } -> tensor<8xf32>
    scf.yield %s : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}
