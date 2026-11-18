//===- bufferize_and_lower.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie
// RUN: air-opt %s -air-lower-linalg-tensors | FileCheck %s

// The linalg.generic is bufferized in place, the to_tensor/to_buffer pair
// around it folds away, and the generic becomes an affine loop.

// CHECK-LABEL: func.func @add
// CHECK-SAME: (%[[A:.*]]: memref<16xf32>, %[[B:.*]]: memref<16xf32>, %[[C:.*]]: memref<16xf32>)
// CHECK-NOT: bufferization.
// CHECK: affine.for %[[I:.*]] = 0 to 16
// CHECK:   affine.load %[[A]][%[[I]]]
// CHECK:   affine.load %[[B]][%[[I]]]
// CHECK:   affine.store %{{.*}}, %[[C]][%[[I]]]
// CHECK-NOT: bufferization.
func.func @add(%a: memref<16xf32>, %b: memref<16xf32>, %c: memref<16xf32>) {
  %ta = bufferization.to_tensor %a restrict : memref<16xf32> to tensor<16xf32>
  %tb = bufferization.to_tensor %b restrict : memref<16xf32> to tensor<16xf32>
  %init = bufferization.to_tensor %c restrict writable : memref<16xf32> to tensor<16xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%ta, %tb : tensor<16xf32>, tensor<16xf32>) outs(%init : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  %m = bufferization.to_buffer %r : tensor<16xf32> to memref<16xf32>
  memref.copy %m, %c : memref<16xf32> to memref<16xf32>
  return
}
