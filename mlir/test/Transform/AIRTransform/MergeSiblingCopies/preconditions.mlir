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

// -----

// Four adjacent copies with a write between the second and the third: the
// merged copy runs where the first one does, so the copies merge in two runs,
// one on each side of the write.
// CHECK-LABEL: @write_between
// CHECK: memref.alloc() : memref<2x16xi16, 1>
// CHECK: memref.copy
// CHECK: memref.store
// CHECK: memref.alloc() : memref<2x16xi16, 1>
// CHECK: memref.copy
// CHECK-NOT: memref.copy
func.func @write_between(%p: memref<*xi16>, %o: index, %q: memref<4xi16>, %x: i16) -> (tensor<2x8xi16>, tensor<2x8xi16>, tensor<2x8xi16>, tensor<2x8xi16>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c24 = arith.constant 24 : index
  %o8 = arith.addi %o, %c8 : index
  %o16 = arith.addi %o, %c16 : index
  %o24 = arith.addi %o, %c24 : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [2, 8], strides: [32, 1] : memref<*xi16> to memref<2x8xi16, strided<[32, 1], offset: ?>>
  %a0 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v0, %a0 : memref<2x8xi16, strided<[32, 1], offset: ?>> to memref<2x8xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  %v1 = memref.reinterpret_cast %p to offset: [%o8], sizes: [2, 8], strides: [32, 1] : memref<*xi16> to memref<2x8xi16, strided<[32, 1], offset: ?>>
  %a1 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v1, %a1 : memref<2x8xi16, strided<[32, 1], offset: ?>> to memref<2x8xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  memref.store %x, %q[%c0] : memref<4xi16>
  %v2 = memref.reinterpret_cast %p to offset: [%o16], sizes: [2, 8], strides: [32, 1] : memref<*xi16> to memref<2x8xi16, strided<[32, 1], offset: ?>>
  %a2 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v2, %a2 : memref<2x8xi16, strided<[32, 1], offset: ?>> to memref<2x8xi16, 1>
  %t2 = bufferization.to_tensor %a2 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  %v3 = memref.reinterpret_cast %p to offset: [%o24], sizes: [2, 8], strides: [32, 1] : memref<*xi16> to memref<2x8xi16, strided<[32, 1], offset: ?>>
  %a3 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v3, %a3 : memref<2x8xi16, strided<[32, 1], offset: ?>> to memref<2x8xi16, 1>
  %t3 = bufferization.to_tensor %a3 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  return %t0, %t1, %t2, %t3 : tensor<2x8xi16>, tensor<2x8xi16>, tensor<2x8xi16>, tensor<2x8xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %m = transform.air.merge_sibling_copies %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A copy of another group between two adjacent copies writes only its own
// buffer: the two still merge.
// CHECK-LABEL: @other_group_between
// CHECK: memref.alloc() : memref<2x16xi16, 1>
// CHECK: memref.copy
// CHECK: memref.copy
// CHECK-NOT: memref.copy
func.func @other_group_between(%p: memref<*xi16>, %r: memref<*xbf16>, %o: index) -> (tensor<2x8xi16>, tensor<4xbf16>, tensor<2x8xi16>) {
  %c8 = arith.constant 8 : index
  %o8 = arith.addi %o, %c8 : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [2, 8], strides: [16, 1] : memref<*xi16> to memref<2x8xi16, strided<[16, 1], offset: ?>>
  %a0 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v0, %a0 : memref<2x8xi16, strided<[16, 1], offset: ?>> to memref<2x8xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  %w = memref.reinterpret_cast %r to offset: [%o], sizes: [4], strides: [1] : memref<*xbf16> to memref<4xbf16, strided<[1], offset: ?>>
  %b = memref.alloc() : memref<4xbf16, 1>
  memref.copy %w, %b : memref<4xbf16, strided<[1], offset: ?>> to memref<4xbf16, 1>
  %tb = bufferization.to_tensor %b restrict writable : memref<4xbf16, 1> to tensor<4xbf16>
  %v1 = memref.reinterpret_cast %p to offset: [%o8], sizes: [2, 8], strides: [16, 1] : memref<*xi16> to memref<2x8xi16, strided<[16, 1], offset: ?>>
  %a1 = memref.alloc() : memref<2x8xi16, 1>
  memref.copy %v1, %a1 : memref<2x8xi16, strided<[16, 1], offset: ?>> to memref<2x8xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<2x8xi16, 1> to tensor<2x8xi16>
  return %t0, %tb, %t1 : tensor<2x8xi16>, tensor<4xbf16>, tensor<2x8xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %m = transform.air.merge_sibling_copies %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
