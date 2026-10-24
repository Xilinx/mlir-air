//===- no_recompute_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/no_recompute.mlir' %s | FileCheck %s
// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s --check-prefix=PLAIN

// With no_recompute a producer is fused only into its only user, read through
// a permutation, and not into a contraction. Without it, all of these fuse.

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

// A contraction stays one.
// CHECK-LABEL: @into_contraction
// CHECK: %[[W:.*]] = linalg.generic
// CHECK: arith.extf
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %[[W]] :
// CHECK-NOT: arith.extf
// CHECK: arith.mulf
// PLAIN-LABEL: @into_contraction
// PLAIN: linalg.generic
// PLAIN: arith.extf
// PLAIN: arith.mulf
// PLAIN-NOT: linalg.generic
func.func @into_contraction(%a: tensor<4x8xf32>, %b: tensor<8x4xbf16>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %w = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%b : tensor<8x4xbf16>) outs(%e : tensor<8x4xf32>) {
  ^bb0(%x: bf16, %o: f32):
    %0 = arith.extf %x : bf16 to f32
    linalg.yield %0 : f32
  } -> tensor<8x4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %w : tensor<4x8xf32>, tensor<8x4xf32>) outs(%c : tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %o, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
