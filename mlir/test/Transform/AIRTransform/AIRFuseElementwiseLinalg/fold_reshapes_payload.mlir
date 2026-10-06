//===- fold_reshapes_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/fold_reshapes.mlir' %s | FileCheck %s
// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s --check-prefix=PLAIN

// Two elementwise generics split by a reshape (the unit dim a Triton
// `x[:, None]` broadcast adds). With fold_reshapes they fuse into one;
// without it the reshape keeps them apart.
// CHECK-LABEL: @split_by_expand
// CHECK: linalg.generic
// CHECK-NOT: linalg.generic
// PLAIN-LABEL: @split_by_expand
// PLAIN: linalg.generic
// PLAIN: tensor.expand_shape
// PLAIN: linalg.generic
#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @split_by_expand(%x: tensor<16xf32>, %y: tensor<16x1xf32>) -> tensor<16x1xf32> {
  %e1 = tensor.empty() : tensor<16xf32>
  %a = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
      ins(%x : tensor<16xf32>) outs(%e1 : tensor<16xf32>) {
  ^bb0(%i: f32, %o: f32):
    %m = arith.mulf %i, %i : f32
    linalg.yield %m : f32
  } -> tensor<16xf32>
  %ae = tensor.expand_shape %a [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  %e2 = tensor.empty() : tensor<16x1xf32>
  %b = linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%ae, %y : tensor<16x1xf32>, tensor<16x1xf32>) outs(%e2 : tensor<16x1xf32>) {
  ^bb0(%i: f32, %j: f32, %o: f32):
    %s = arith.addf %i, %j : f32
    linalg.yield %s : f32
  } -> tensor<16x1xf32>
  return %b : tensor<16x1xf32>
}

// A reshape feeding a contraction is left alone: only elementwise chains fuse.
// CHECK-LABEL: @reshape_into_matmul
// CHECK: tensor.collapse_shape
// CHECK: linalg.matmul
func.func @reshape_into_matmul(%x: tensor<4x4x8xf32>, %w: tensor<8x8xf32>) -> tensor<16x8xf32> {
  %e1 = tensor.empty() : tensor<4x4x8xf32>
  %a = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<4x4x8xf32>) outs(%e1 : tensor<4x4x8xf32>) {
  ^bb0(%i: f32, %o: f32):
    %m = arith.mulf %i, %i : f32
    linalg.yield %m : f32
  } -> tensor<4x4x8xf32>
  %ac = tensor.collapse_shape %a [[0, 1], [2]] : tensor<4x4x8xf32> into tensor<16x8xf32>
  %z = arith.constant 0.0 : f32
  %e2 = tensor.empty() : tensor<16x8xf32>
  %f = linalg.fill ins(%z : f32) outs(%e2 : tensor<16x8xf32>) -> tensor<16x8xf32>
  %r = linalg.matmul ins(%ac, %w : tensor<16x8xf32>, tensor<8x8xf32>) outs(%f : tensor<16x8xf32>) -> tensor<16x8xf32>
  return %r : tensor<16x8xf32>
}
