//===- channel_tiling.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter %s | FileCheck %s

// A channel op without explicit sizes iterates over the whole memref: static
// dims give constant loop bounds, dynamic dims a memref.dim.

// CHECK-LABEL: func.func @static_shape
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C64]] step
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C32]] step
// CHECK: air.channel.put @chan[]
func.func @static_shape(%m: memref<64x32xi32>) {
  air.channel.put @chan[] (%m[] [] []) : (memref<64x32xi32>)
  return
}

// CHECK-LABEL: func.func @dynamic_shape
// CHECK-SAME: %[[M:[^:]*]]: memref<?x32xi32>
// CHECK-DAG: %[[DIM:.*]] = memref.dim %[[M]], %{{.*}} : memref<?x32xi32>
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[DIM]] step
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C32]] step
// CHECK: air.channel.put @chan[]
func.func @dynamic_shape(%m: memref<?x32xi32>) {
  air.channel.put @chan[] (%m[] [] []) : (memref<?x32xi32>)
  return
}

air.channel @chan [1, 1]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %puts = transform.structured.match ops{["air.channel.put"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %puts tile_sizes [16, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
