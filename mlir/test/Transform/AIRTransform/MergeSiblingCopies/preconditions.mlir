//===- preconditions.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// [2, 8] at o, o + 8 and [1, 8] at o + 32, rows 16 apart: the bounding region
// is 3 x 16, larger than the copies and ending past the last of them, so the
// copies stay.
// CHECK-LABEL: @box_past_members
// CHECK-COUNT-3: memref.copy
func.func @box_past_members(%p: memref<*xi16>, %o: index) -> (tensor<2x8xi16>, tensor<2x8xi16>, tensor<1x8xi16>) {
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %o8 = arith.addi %o, %c8 : index
  %o32 = arith.addi %o, %c32 : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [2, 8], strides: [16, 1] : memref<*xi16> to memref<2x8xi16, strided<[16, 1], offset: ?>>
  %a0 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v0, %a0 : memref<2x8xi16, strided<[16, 1], offset: ?>> to memref<2x8xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  %v1 = memref.reinterpret_cast %p to offset: [%o8], sizes: [2, 8], strides: [16, 1] : memref<*xi16> to memref<2x8xi16, strided<[16, 1], offset: ?>>
  %a1 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v1, %a1 : memref<2x8xi16, strided<[16, 1], offset: ?>> to memref<2x8xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  %v2 = memref.reinterpret_cast %p to offset: [%o32], sizes: [1, 8], strides: [16, 1] : memref<*xi16> to memref<1x8xi16, strided<[16, 1], offset: ?>>
  %a2 = memref.alloc() : memref<1x8xi16, 1>
  memref.copy %v2, %a2 : memref<1x8xi16, strided<[16, 1], offset: ?>> to memref<1x8xi16, 1>
  %t2 = bufferization.to_tensor %a2 restrict writable : memref<1x8xi16, 1> to tensor<1x8xi16>
  return %t0, %t1, %t2 : tensor<2x8xi16>, tensor<2x8xi16>, tensor<1x8xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %m = transform.air.merge_sibling_copies %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Two small copies far apart would become one large one: they stay.
// CHECK-LABEL: @far_apart
// CHECK-COUNT-2: memref.copy
func.func @far_apart(%p: memref<*xi16>, %o: index) -> (tensor<4xi16>, tensor<4xi16>) {
  %cfar = arith.constant 1048576 : index
  %of = arith.addi %o, %cfar : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a0 = memref.alloc() : memref<4xi16, 1>
  memref.copy %v0, %a0 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<4xi16, 1> to tensor<4xi16>
  %v1 = memref.reinterpret_cast %p to offset: [%of], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a1 = memref.alloc() : memref<4xi16, 1>
  memref.copy %v1, %a1 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<4xi16, 1> to tensor<4xi16>
  return %t0, %t1 : tensor<4xi16>, tensor<4xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %m = transform.air.merge_sibling_copies %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Adjacent copies whose union is the region are merged.
// CHECK-LABEL: @adjacent
// CHECK: memref.alloc() : memref<2x16xi16, 1>
// CHECK-COUNT-1: memref.copy
// CHECK-NOT: memref.copy
func.func @adjacent(%p: memref<*xi16>, %o: index) -> (tensor<2x8xi16>, tensor<2x8xi16>) {
  %c8 = arith.constant 8 : index
  %o8 = arith.addi %o, %c8 : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [2, 8], strides: [16, 1] : memref<*xi16> to memref<2x8xi16, strided<[16, 1], offset: ?>>
  %a0 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v0, %a0 : memref<2x8xi16, strided<[16, 1], offset: ?>> to memref<2x8xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  %v1 = memref.reinterpret_cast %p to offset: [%o8], sizes: [2, 8], strides: [16, 1] : memref<*xi16> to memref<2x8xi16, strided<[16, 1], offset: ?>>
  %a1 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v1, %a1 : memref<2x8xi16, strided<[16, 1], offset: ?>> to memref<2x8xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  return %t0, %t1 : tensor<2x8xi16>, tensor<2x8xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %m = transform.air.merge_sibling_copies %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
