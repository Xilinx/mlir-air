//===- tensor_linalg_generic.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Test that air-dependency correctly handles tensor-mode linalg.generic ops
// whose tensor results are consumed by bufferization.materialize_in_destination.
// The linalg.generic should be wrapped in an air.execute that forwards the
// tensor result (issue #1369).

// CHECK-LABEL: @tensor_linalg_materialize
// CHECK: %[[TOKEN:.*]], %[[RESULT:.*]] = air.execute
// CHECK:   %[[GENERIC:.*]] = linalg.generic
// CHECK:   air.execute_terminator %[[GENERIC]]
// CHECK: air.execute
// CHECK:   bufferization.materialize_in_destination %[[RESULT]]

#map = affine_map<(d0) -> (d0)>

module {
  func.func @tensor_linalg_materialize(%arg0: memref<64xf32>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%out=%arg0) : memref<64xf32> {
      %cst = arith.constant 2.000000e+00 : f32
      %alloc = memref.alloc() : memref<64xf32, 2>
      %input = bufferization.to_tensor %alloc restrict writable : memref<64xf32, 2> to tensor<64xf32>
      %empty = tensor.empty() : tensor<64xf32>
      %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%input : tensor<64xf32>) outs(%empty : tensor<64xf32>) {
      ^bb0(%in: f32, %o: f32):
        %mul = arith.mulf %in, %cst : f32
        linalg.yield %mul : f32
      } -> tensor<64xf32>
      bufferization.materialize_in_destination %result in writable %out : (tensor<64xf32>, memref<64xf32>) -> ()
      memref.dealloc %alloc : memref<64xf32, 2>
    }
    return
  }
}
