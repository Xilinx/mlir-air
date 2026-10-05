//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// CHECK-LABEL: @step8
// CHECK: scf.for %[[J:.*]] = %c0{{.*}} to %c32{{.*}} step %c1
// CHECK: %[[I:.*]] = affine.apply #{{.*}}(%[[J]])
// CHECK: tensor.extract_slice %arg0[%[[I]]] [8] [1]
func.func @step8(%t: tensor<256xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %r = scf.for %i = %c0 to %c256 step %c8 iter_args(%acc = %init) -> (tensor<8xf32>) {
    %s = tensor.extract_slice %t[%i] [8] [1] : tensor<256xf32> to tensor<8xf32>
    %a = arith.addf %acc, %s : tensor<8xf32>
    scf.yield %a : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}
