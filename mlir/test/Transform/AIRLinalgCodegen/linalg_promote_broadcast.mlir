//===- linalg_promote_broadcast.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for https://github.com/Xilinx/mlir-air/issues/1403
// Verify that linalg_promote promotes operands with broadcast indexing maps
// (non-subview operands) to the target memory space.

// RUN: air-opt %s -air-transform='filename=%s' -split-input-file | FileCheck %s

#map_identity = affine_map<(d0, d1) -> (d0, d1)>
#map_broadcast = affine_map<(d0, d1) -> (d1)>

// Test A: Mixed subview and non-subview (broadcast) operands.
// The broadcast input (weight) should be promoted to L1 alongside the subview
// input and output.
//
// CHECK-LABEL: func.func @broadcast_promote_mixed
// CHECK: memref.alloc() : memref<64xf32, 2 : i32>
// CHECK: memref.copy {{.*}} : memref<64xf32> to memref<64xf32, 2 : i32>
// CHECK: linalg.generic
// CHECK-SAME: memref<64xf32, 2 : i32>
func.func @broadcast_promote_mixed(
    %input: memref<16x64xf32>,
    %weight: memref<64xf32>,
    %output: memref<16x64xf32>) {
  %c0 = arith.constant 0 : index
  %subview_in = memref.subview %input[%c0, 0] [4, 64] [1, 1]
      : memref<16x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
  %subview_out = memref.subview %output[%c0, 0] [4, 64] [1, 1]
      : memref<16x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
  linalg.generic {
    indexing_maps = [#map_identity, #map_broadcast, #map_identity],
    iterator_types = ["parallel", "parallel"]
  } ins(%subview_in, %weight
      : memref<4x64xf32, strided<[64, 1], offset: ?>>, memref<64xf32>)
    outs(%subview_out
      : memref<4x64xf32, strided<[64, 1], offset: ?>>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %mul = arith.mulf %in0, %in1 : f32
    linalg.yield %mul : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.air.linalg_promote %0 {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map_id2 = affine_map<(d0, d1) -> (d0, d1)>
#map_bc2 = affine_map<(d0, d1) -> (d1)>

// Test B: All non-subview operands (none are subviews).
// All operands should be promoted to L1.
//
// CHECK-LABEL: func.func @broadcast_promote_all_non_subview
// CHECK: memref.alloc() : memref<32xf32, 2 : i32>
// CHECK: memref.copy {{.*}} : memref<32xf32, 1> to memref<32xf32, 2 : i32>
// CHECK: memref.alloc() : memref<4x32xf32, 2 : i32>
// CHECK: memref.copy {{.*}} : memref<4x32xf32, 1> to memref<4x32xf32, 2 : i32>
// CHECK: linalg.generic
// CHECK-SAME: memref<32xf32, 2 : i32>
// CHECK-SAME: memref<4x32xf32, 2 : i32>
// CHECK: memref.copy {{.*}} : memref<4x32xf32, 2 : i32> to memref<4x32xf32, 1>
func.func @broadcast_promote_all_non_subview(
    %weight: memref<32xf32, 1>,
    %output: memref<4x32xf32, 1>) {
  linalg.generic {
    indexing_maps = [#map_bc2, #map_id2],
    iterator_types = ["parallel", "parallel"]
  } ins(%weight : memref<32xf32, 1>)
    outs(%output : memref<4x32xf32, 1>) {
  ^bb0(%in0: f32, %out: f32):
    %add = arith.addf %in0, %out : f32
    linalg.yield %add : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.air.linalg_promote %0 {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
