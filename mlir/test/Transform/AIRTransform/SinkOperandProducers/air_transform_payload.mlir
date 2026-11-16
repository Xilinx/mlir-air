//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// The lhs's slice and transpose move to right before the generic, after the
// rhs's, in their order.
// CHECK-LABEL: @lhs_last
// CHECK: %[[B:.*]] = tensor.extract_slice %arg1
// CHECK: %[[BT:.*]] = linalg.transpose ins(%[[B]]
// CHECK: %[[A:.*]] = tensor.extract_slice %arg0
// CHECK: %[[AT:.*]] = linalg.transpose ins(%[[A]]
// CHECK: linalg.generic {{.*}} ins(%[[AT]], %[[BT]]
func.func @lhs_last(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %i: index) -> tensor<8x8xf32> {
  %as = tensor.extract_slice %a[%i, 0] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e0 = tensor.empty() : tensor<8x8xf32>
  %at = linalg.transpose ins(%as : tensor<8x8xf32>) outs(%e0 : tensor<8x8xf32>) permutation = [1, 0]
  %bs = tensor.extract_slice %b[0, %i] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e1 = tensor.empty() : tensor<8x8xf32>
  %bt = linalg.transpose ins(%bs : tensor<8x8xf32>) outs(%e1 : tensor<8x8xf32>) permutation = [1, 0]
  %e2 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%at, %bt : tensor<8x8xf32>, tensor<8x8xf32>) outs(%e2 : tensor<8x8xf32>) attrs = {sink} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// A producer whose result is used elsewhere stays, and so do the ops it
// uses; only the transpose, used by the generic alone, moves.
// CHECK-LABEL: @used_elsewhere
// CHECK: %[[A:.*]] = tensor.extract_slice %arg0
// CHECK: %[[B:.*]] = tensor.extract_slice %arg1
// CHECK: %[[AT:.*]] = linalg.transpose ins(%[[A]]
// CHECK: linalg.generic {{.*}} ins(%[[AT]], %[[B]]
// CHECK: return {{.*}}, %[[A]]
func.func @used_elsewhere(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %i: index) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %as = tensor.extract_slice %a[%i, 0] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e0 = tensor.empty() : tensor<8x8xf32>
  %at = linalg.transpose ins(%as : tensor<8x8xf32>) outs(%e0 : tensor<8x8xf32>) permutation = [1, 0]
  %bs = tensor.extract_slice %b[0, %i] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e2 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%at, %bs : tensor<8x8xf32>, tensor<8x8xf32>) outs(%e2 : tensor<8x8xf32>) attrs = {sink} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %r, %as : tensor<8x8xf32>, tensor<8x8xf32>
}

// A producer with memory effects (the load) stays where it is, and so does
// the index it uses; the splat of the loaded value moves.
// CHECK-LABEL: @effects
// CHECK: %[[I:.*]] = arith.addi %arg2
// CHECK: %[[V:.*]] = memref.load %arg0[%[[I]]]
// CHECK: %[[B:.*]] = tensor.extract_slice %arg1
// CHECK: %[[T:.*]] = tensor.splat %[[V]]
// CHECK: linalg.generic {{.*}} ins(%[[T]], %[[B]]
func.func @effects(%m: memref<64xf32>, %b: tensor<64x64xf32>, %i: index) -> tensor<8x8xf32> {
  %c1 = arith.constant 1 : index
  %j = arith.addi %i, %c1 : index
  %v = memref.load %m[%j] : memref<64xf32>
  %t = tensor.splat %v : tensor<8x8xf32>
  %bs = tensor.extract_slice %b[0, %i] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e2 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%t, %bs : tensor<8x8xf32>, tensor<8x8xf32>) outs(%e2 : tensor<8x8xf32>) attrs = {sink} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}

// The transpose stays (its result is used elsewhere), so the slice it reads
// stays before it, though only the transpose uses the slice.
// CHECK-LABEL: @kept_uses
// CHECK: %[[A:.*]] = tensor.extract_slice %arg0
// CHECK: %[[AT:.*]] = linalg.transpose ins(%[[A]]
// CHECK: %[[B:.*]] = tensor.extract_slice %arg1
// CHECK: linalg.generic {{.*}} ins(%[[AT]], %[[B]]
func.func @kept_uses(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %i: index) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %as = tensor.extract_slice %a[%i, 0] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e0 = tensor.empty() : tensor<8x8xf32>
  %at = linalg.transpose ins(%as : tensor<8x8xf32>) outs(%e0 : tensor<8x8xf32>) permutation = [1, 0]
  %bs = tensor.extract_slice %b[0, %i] [8, 8] [1, 1] : tensor<64x64xf32> to tensor<8x8xf32>
  %e2 = tensor.empty() : tensor<8x8xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%at, %bs : tensor<8x8xf32>, tensor<8x8xf32>) outs(%e2 : tensor<8x8xf32>) attrs = {sink} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %r, %at : tensor<8x8xf32>, tensor<8x8xf32>
}
