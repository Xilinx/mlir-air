//===- pack_basic.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-matmul-pack-and-transpose='pack-sizes=8,8,8' \
// RUN:   | FileCheck %s --check-prefix=NOPERM
// RUN: air-opt %s -air-matmul-pack-and-transpose='pack-sizes=8,8,8 \
// RUN:   lhs-outer-perm=1,0 rhs-outer-perm=1,0 rhs-inner-perm=1,0 \
// RUN:   acc-outer-perm=1,0' \
// RUN:   | FileCheck %s --check-prefix=ALLPERM

// NOPERM-LABEL: func.func @matmul_pack_basic
// NOPERM:       linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 8]
// NOPERM:       linalg.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [8, 8]
// NOPERM:       linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 8]
// NOPERM:       linalg.generic
// NOPERM-SAME:    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// NOPERM-SAME:    packed_matmul
// NOPERM:       linalg.unpack

// Test 54-style transposes: outer_perm=[1,0] on LHS, RHS, ACC + inner_perm=[1,0] on RHS.
// LHS (M,K) → outer-transposed to (K,M).
// RHS originally inner_dims_pos=[1,0]; outer_perm + inner_perm both [1,0] → inner_dims_pos=[0,1].
// ACC outer-transposed (M,N) → (N,M).
// ALLPERM-LABEL: func.func @matmul_pack_basic
// ALLPERM:       linalg.pack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8]
// ALLPERM:       linalg.pack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8]
// ALLPERM:       linalg.pack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8]
// ALLPERM:       linalg.generic
// ALLPERM-SAME:    packed_matmul
// ALLPERM:       linalg.unpack %{{.*}} outer_dims_perm = [1, 0]

func.func @matmul_pack_basic(%a: tensor<256x784xf32>, %b: tensor<784x128xf32>) -> tensor<256x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<256x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x128xf32>) -> tensor<256x128xf32>
  %2 = linalg.matmul ins(%a, %b : tensor<256x784xf32>, tensor<784x128xf32>) outs(%1 : tensor<256x128xf32>) -> tensor<256x128xf32>
  return %2 : tensor<256x128xf32>
}
