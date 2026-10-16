//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// The transpose moves above the collapse, moving each group's dims together.
// CHECK-LABEL: @swap
// CHECK: %[[T:.*]] = linalg.transpose ins(%arg0 : tensor<4x2x8xf32>) outs(%{{.*}} : tensor<8x4x2xf32>) permutation = [2, 0, 1]
// CHECK: %[[C:.*]] = tensor.collapse_shape %[[T]] {{\[}}[0], [1, 2]] : tensor<8x4x2xf32> into tensor<8x8xf32>
// CHECK: return %[[C]]
func.func @swap(%x: tensor<4x2x8xf32>) -> tensor<8x8xf32> {
  %c = tensor.collapse_shape %x [[0, 1], [2]] : tensor<4x2x8xf32> into tensor<8x8xf32>
  %e = tensor.empty() : tensor<8x8xf32>
  %t = linalg.transpose ins(%c : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) permutation = [1, 0]
  return %t : tensor<8x8xf32>
}

// Two transposes that a collapse kept apart meet and cancel.
// CHECK-LABEL: @cancel
// CHECK-NOT: linalg.transpose
// CHECK: %[[C:.*]] = tensor.collapse_shape %arg0 {{\[}}[0, 1], [2]] : tensor<2x3x4xf32> into tensor<6x4xf32>
// CHECK: return %[[C]]
func.func @cancel(%x: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %e0 = tensor.empty() : tensor<4x2x3xf32>
  %t0 = linalg.transpose ins(%x : tensor<2x3x4xf32>) outs(%e0 : tensor<4x2x3xf32>) permutation = [2, 0, 1]
  %c = tensor.collapse_shape %t0 [[0], [1, 2]] : tensor<4x2x3xf32> into tensor<4x6xf32>
  %e1 = tensor.empty() : tensor<6x4xf32>
  %t1 = linalg.transpose ins(%c : tensor<4x6xf32>) outs(%e1 : tensor<6x4xf32>) permutation = [1, 0]
  return %t1 : tensor<6x4xf32>
}

// A dynamic source is left alone.
// CHECK-LABEL: @dynamic
// CHECK: tensor.collapse_shape %arg0
// CHECK: linalg.transpose
func.func @dynamic(%x: tensor<?x2x8xf32>, %n: index) -> tensor<8x?xf32> {
  %c = tensor.collapse_shape %x [[0, 1], [2]] : tensor<?x2x8xf32> into tensor<?x8xf32>
  %e = tensor.empty(%n) : tensor<8x?xf32>
  %t = linalg.transpose ins(%c : tensor<?x8xf32>) outs(%e : tensor<8x?xf32>) permutation = [1, 0]
  return %t : tensor<8x?xf32>
}
