//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// A packed accumulator [N/8, M/8, 8, 8] whose unpacked columns are interleaved
// pairs ([M, N/2, 2]) feeding one elementwise op: each pair slot
// becomes a slice of the packed tile dim split [4, 2], and the elementwise op
// runs on the packed slices with one unpack (inner tile 8 x 4) after it.

// CHECK-LABEL: @pairs
// CHECK: %[[E:.*]] = tensor.expand_shape %arg0 {{\[}}[0], [1], [2], [3, 4]] output_shape [4, 2, 8, 4, 2] : tensor<4x2x8x8xf32> into tensor<4x2x8x4x2xf32>
// CHECK-DAG: %[[G:.*]] = tensor.extract_slice %[[E]][0, 0, 0, 0, 0] [4, 2, 8, 4, 1] [1, 1, 1, 1, 1] : tensor<4x2x8x4x2xf32> to tensor<4x2x8x4xf32>
// CHECK-DAG: %[[U:.*]] = tensor.extract_slice %[[E]][0, 0, 0, 0, 1] [4, 2, 8, 4, 1] [1, 1, 1, 1, 1] : tensor<4x2x8x4x2xf32> to tensor<4x2x8x4xf32>
// CHECK: %[[P:.*]] = linalg.generic {{.*}} ins(%[[G]], %[[U]] : tensor<4x2x8x4xf32>, tensor<4x2x8x4xf32>) outs(%{{.*}} : tensor<4x2x8x4xbf16>) attrs = {keep}
// CHECK: %[[R:.*]] = linalg.unpack %[[P]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %{{.*}} : tensor<4x2x8x4xbf16> -> tensor<16x16xbf16>
// CHECK: return %[[R]]
#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @pairs(%acc: tensor<4x2x8x8xf32>) -> tensor<16x16xbf16> {
  %e0 = tensor.empty() : tensor<16x32xf32>
  %u = linalg.unpack %acc outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e0 : tensor<4x2x8x8xf32> -> tensor<16x32xf32>
  %x = tensor.expand_shape %u [[0], [1, 2]] output_shape [16, 16, 2] : tensor<16x32xf32> into tensor<16x16x2xf32>
  %g = tensor.extract_slice %x[0, 0, 0] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %v = tensor.extract_slice %x[0, 0, 1] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %o = tensor.empty() : tensor<16x16xbf16>
  %h = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%g, %v : tensor<16x16xf32>, tensor<16x16xf32>) outs(%o : tensor<16x16xbf16>) attrs = {keep} {
  ^bb0(%a: f32, %b: f32, %c: bf16):
    %m = arith.mulf %a, %b : f32
    %t = arith.truncf %m : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<16x16xbf16>
  return %h : tensor<16x16xbf16>
}

// The elementwise op also reads a tensor that is not one of the unpacked
// slices. The unpack still moves below the slice, but the elementwise op stays
// on unpacked values: only an op whose inputs are all such unpacks can run on
// the packed layout.
// CHECK-LABEL: @mixed_inputs
// CHECK: tensor.expand_shape %arg0
// CHECK: %[[S:.*]] = tensor.extract_slice
// CHECK: %[[U:.*]] = linalg.unpack %[[S]]
// CHECK: linalg.generic {{.*}} ins(%[[U]], %arg1 : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NOT: linalg.unpack
func.func @mixed_inputs(%acc: tensor<4x2x8x8xf32>, %other: tensor<16x16xf32>) -> tensor<16x16xbf16> {
  %e0 = tensor.empty() : tensor<16x32xf32>
  %u = linalg.unpack %acc outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %e0 : tensor<4x2x8x8xf32> -> tensor<16x32xf32>
  %x = tensor.expand_shape %u [[0], [1, 2]] output_shape [16, 16, 2] : tensor<16x32xf32> into tensor<16x16x2xf32>
  %g = tensor.extract_slice %x[0, 0, 0] [16, 16, 1] [1, 1, 1] : tensor<16x16x2xf32> to tensor<16x16xf32>
  %o = tensor.empty() : tensor<16x16xbf16>
  %h = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%g, %other : tensor<16x16xf32>, tensor<16x16xf32>) outs(%o : tensor<16x16xbf16>) {
  ^bb0(%a: f32, %b: f32, %c: bf16):
    %m = arith.mulf %a, %b : f32
    %t = arith.truncf %m : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<16x16xbf16>
  return %h : tensor<16x16xbf16>
}
