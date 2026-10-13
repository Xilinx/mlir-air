//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// One pack feeding two inputs is folded into both and erased once.
// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 * 8 + d2, d0 * 8 + d3)>
// CHECK-LABEL: @pack_twice
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], {{.*}}]{{.*}} ins(%arg0, %arg0 : tensor<16x16xbf16>, tensor<16x16xbf16>)
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @pack_twice(%src: tensor<16x16xbf16>) -> tensor<2x2x8x8xbf16> {
  %e = tensor.empty() : tensor<2x2x8x8xbf16>
  %p = linalg.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e : tensor<16x16xbf16> -> tensor<2x2x8x8xbf16>
  %init = tensor.empty() : tensor<2x2x8x8xbf16>
  %r = linalg.generic {indexing_maps = [#id4, #id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%p, %p : tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) outs(%init : tensor<2x2x8x8xbf16>) {
  ^bb0(%a: bf16, %b: bf16, %o: bf16):
    %x = arith.addf %a, %b : bf16
    linalg.yield %x : bf16
  } -> tensor<2x2x8x8xbf16>
  return %r : tensor<2x2x8x8xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.fold_pack_into_generic %g : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
