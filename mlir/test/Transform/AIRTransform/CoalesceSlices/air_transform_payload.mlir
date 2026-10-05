//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Three slices of one tile (values, scale row, offset row) at a common column
// offset become slices of one copy of their bounding region.
// CHECK-LABEL: @one_copy
// CHECK: %[[R:.*]] = tensor.extract_slice %arg0[0, 0, %arg1] [2, 10, 8] [1, 1, 1]
// CHECK: %[[E:.*]] = tensor.empty() : tensor<2x10x8xi16>
// CHECK: %[[C:.*]] = linalg.copy ins(%[[R]] : tensor<2x10x8xi16>) outs(%[[E]] : tensor<2x10x8xi16>)
// CHECK-DAG: tensor.extract_slice %[[C]][0, 0, 0] [2, 8, 8] [1, 1, 1]
// CHECK-DAG: tensor.extract_slice %[[C]][0, 8, 0] [2, 1, 8] [1, 1, 1]
// CHECK-DAG: tensor.extract_slice %[[C]][0, 9, 0] [2, 1, 8] [1, 1, 1]
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#bc = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
func.func @one_copy(%t: tensor<2x10x64xi16>, %n: index) -> tensor<2x8x8xi16> {
  %q = tensor.extract_slice %t[0, 0, %n] [2, 8, 8] [1, 1, 1] : tensor<2x10x64xi16> to tensor<2x8x8xi16>
  %s = tensor.extract_slice %t[0, 8, %n] [2, 1, 8] [1, 1, 1] : tensor<2x10x64xi16> to tensor<2x1x8xi16>
  %m = tensor.extract_slice %t[0, 9, %n] [2, 1, 8] [1, 1, 1] : tensor<2x10x64xi16> to tensor<2x1x8xi16>
  %init = tensor.empty() : tensor<2x8x8xi16>
  %r = linalg.generic {indexing_maps = [#id, #bc, #bc, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%q, %s, %m : tensor<2x8x8xi16>, tensor<2x1x8xi16>, tensor<2x1x8xi16>) outs(%init : tensor<2x8x8xi16>) {
  ^bb0(%a: i16, %b: i16, %c: i16, %o: i16):
    %x = arith.muli %a, %b : i16
    %y = arith.addi %x, %c : i16
    linalg.yield %y : i16
  } -> tensor<2x8x8xi16>
  return %r : tensor<2x8x8xi16>
}

// Slices reaching the inputs through an expand_shape (a unit dim added by
// broadcasting) are coalesced too; the expand stays on the new slice.
// CHECK-LABEL: @through_expand
// CHECK: %[[C:.*]] = linalg.copy
// CHECK: %[[S:.*]] = tensor.extract_slice %[[C]][0, 8, 0] [2, 1, 8] [1, 1, 1] : tensor<2x9x8xi16> to tensor<2x8xi16>
// CHECK: tensor.expand_shape %[[S]]
func.func @through_expand(%t: tensor<2x10x64xi16>, %n: index) -> tensor<2x8x8xi16> {
  %q = tensor.extract_slice %t[0, 0, %n] [2, 8, 8] [1, 1, 1] : tensor<2x10x64xi16> to tensor<2x8x8xi16>
  %s2 = tensor.extract_slice %t[0, 8, %n] [2, 1, 8] [1, 1, 1] : tensor<2x10x64xi16> to tensor<2x8xi16>
  %s = tensor.expand_shape %s2 [[0, 1], [2]] output_shape [2, 1, 8] : tensor<2x8xi16> into tensor<2x1x8xi16>
  %init = tensor.empty() : tensor<2x8x8xi16>
  %r = linalg.generic {indexing_maps = [#id, #bc, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%q, %s : tensor<2x8x8xi16>, tensor<2x1x8xi16>) outs(%init : tensor<2x8x8xi16>) {
  ^bb0(%a: i16, %b: i16, %o: i16):
    %x = arith.muli %a, %b : i16
    linalg.yield %x : i16
  } -> tensor<2x8x8xi16>
  return %r : tensor<2x8x8xi16>
}
