//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// A generic reading a packed 16x16 tile reads the unpacked tile through its
// indexing map instead, and the pack goes away.
// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 * 8 + d2, d0 * 8 + d3)>
// CHECK-LABEL: @fold
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], {{.*}}]{{.*}} ins(%arg0 : tensor<16x16xbf16>)
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @fold(%src: tensor<16x16xbf16>) -> tensor<2x2x8x8xbf16> {
  %e = tensor.empty() : tensor<2x2x8x8xbf16>
  %p = linalg.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xbf16> -> tensor<2x2x8x8xbf16>
  %init = tensor.empty() : tensor<2x2x8x8xbf16>
  %r = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p : tensor<2x2x8x8xbf16>) outs(%init : tensor<2x2x8x8xbf16>) {
  ^bb0(%a: bf16, %o: bf16):
    %x = arith.addf %a, %a : bf16
    linalg.yield %x : bf16
  } -> tensor<2x2x8x8xbf16>
  return %r : tensor<2x2x8x8xbf16>
}

// A pack that pads (14 does not divide into tiles of 8) is kept.
// CHECK-LABEL: @padded
// CHECK: linalg.pack
func.func @padded(%src: tensor<14x16xbf16>) -> tensor<2x2x8x8xbf16> {
  %pad = arith.constant 0.0 : bf16
  %e = tensor.empty() : tensor<2x2x8x8xbf16>
  %p = linalg.pack %src padding_value(%pad : bf16) inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<14x16xbf16> -> tensor<2x2x8x8xbf16>
  %init = tensor.empty() : tensor<2x2x8x8xbf16>
  %r = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p : tensor<2x2x8x8xbf16>) outs(%init : tensor<2x2x8x8xbf16>) {
  ^bb0(%a: bf16, %o: bf16):
    linalg.yield %a : bf16
  } -> tensor<2x2x8x8xbf16>
  return %r : tensor<2x2x8x8xbf16>
}
