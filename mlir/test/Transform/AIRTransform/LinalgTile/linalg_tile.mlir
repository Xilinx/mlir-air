//===- linalg_tile.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// Static sizes: a 0 leaves that loop untiled.

// CHECK-LABEL: func.func @static_sizes
// CHECK: scf.forall (%{{.*}}, %{{.*}}) in (4, 4)
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<32x64xf32>, tensor<64x16xf32>)
func.func @static_sizes(%a: tensor<128x64xf32>, %b: tensor<64x64xf32>, %c: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %r = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x64xf32>) outs(%c : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %r : tensor<128x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.air.linalg_tile %matmul [32, 16, 0]
    transform.yield
  }
}

// -----

// A size given by a handle is not known to divide the loop, so it is
// rejected.

func.func @dynamic_size(%a: tensor<128x64xf32>, %b: tensor<64x64xf32>, %c: tensor<128x64xf32>) -> tensor<128x64xf32> {
  // expected-note @below {{target op}}
  %r = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x64xf32>) outs(%c : tensor<128x64xf32>) -> tensor<128x64xf32>
  %c16 = arith.constant {tile_size} 16 : index
  return %r : tensor<128x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %sz = transform.structured.match attributes{tile_size} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{only support tiling in integer factors}}
    %tiled, %loop = transform.air.linalg_tile %matmul [32, %sz, 0]
    transform.yield
  }
}
