//===- raise_load_buffer_to_tensor.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// A masked load with a constant mask, as triton-shared lowers it: the other
// value filled, the in-mask box copied over it.

// CHECK-LABEL: func.func @constant_mask
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<40x96xf32>
// CHECK: memref.copy %{{.*}}, %[[BUF]]
// CHECK: %[[E:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK: %[[F:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[E]] : tensor<64x128xf32>)
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[BUF]] restrict writable : memref<40x96xf32> to tensor<40x96xf32>
// CHECK: %[[R:.*]] = tensor.insert_slice %[[T]] into %[[F]][0, 0] [40, 96] [1, 1]
// CHECK: return %[[R]]
func.func @constant_mask(%src: memref<64x128xf32, strided<[128, 1]>>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %buf = memref.alloc() : memref<64x128xf32>
  linalg.fill ins(%cst : f32) outs(%buf : memref<64x128xf32>)
  %s = memref.subview %src[0, 0] [40, 96] [1, 1] : memref<64x128xf32, strided<[128, 1]>> to memref<40x96xf32, strided<[128, 1]>>
  %d = memref.subview %buf[0, 0] [40, 96] [1, 1] : memref<64x128xf32> to memref<40x96xf32, strided<[128, 1]>>
  memref.copy %s, %d : memref<40x96xf32, strided<[128, 1]>> to memref<40x96xf32, strided<[128, 1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<64x128xf32> to tensor<64x128xf32>
  return %t : tensor<64x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A bound known only at run time: the fill sits under a condition, the box is
// dynamic. The fill is raised as if it ran.

// CHECK-LABEL: func.func @dynamic_mask
// CHECK-NOT: scf.if
// CHECK: %[[BUF:.*]] = memref.alloc(%[[N:.*]]) : memref<64x?xf32>
// CHECK: memref.copy %{{.*}}, %[[BUF]]
// CHECK: %[[F:.*]] = linalg.fill {{.*}} -> tensor<64x128xf32>
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[BUF]] restrict writable : memref<64x?xf32> to tensor<64x?xf32>
// CHECK: tensor.insert_slice %[[T]] into %[[F]][0, 0] [64, %[[N]]] [1, 1]
func.func @dynamic_mask(%src: memref<64x128xf32, strided<[300, 1], offset: ?>>, %n: index) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %buf = memref.alloc() : memref<64x128xf32>
  %cut = arith.cmpi slt, %n, %c128 : index
  scf.if %cut {
    linalg.fill ins(%cst : f32) outs(%buf : memref<64x128xf32>)
  }
  %s = memref.subview %src[0, 0] [64, %n] [1, 1] : memref<64x128xf32, strided<[300, 1], offset: ?>> to memref<64x?xf32, strided<[300, 1], offset: ?>>
  %d = memref.subview %buf[0, 0] [64, %n] [1, 1] : memref<64x128xf32> to memref<64x?xf32, strided<[128, 1]>>
  memref.copy %s, %d : memref<64x?xf32, strided<[300, 1], offset: ?>> to memref<64x?xf32, strided<[128, 1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<64x128xf32> to tensor<64x128xf32>
  return %t : tensor<64x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// No other value: the elements outside the box are uninitialized.

// CHECK-LABEL: func.func @no_other
// CHECK: %[[E:.*]] = tensor.empty() : tensor<64x128xf32>
// CHECK-NOT: linalg.fill
// CHECK: tensor.insert_slice %{{.*}} into %[[E]][0, 0] [64, 96] [1, 1]
func.func @no_other(%src: memref<64x128xf32, strided<[128, 1]>>) -> tensor<64x128xf32> {
  %buf = memref.alloc() : memref<64x128xf32>
  %s = memref.subview %src[0, 0] [64, 96] [1, 1] : memref<64x128xf32, strided<[128, 1]>> to memref<64x96xf32, strided<[128, 1]>>
  %d = memref.subview %buf[0, 0] [64, 96] [1, 1] : memref<64x128xf32> to memref<64x96xf32, strided<[128, 1]>>
  memref.copy %s, %d : memref<64x96xf32, strided<[128, 1]>> to memref<64x96xf32, strided<[128, 1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<64x128xf32> to tensor<64x128xf32>
  return %t : tensor<64x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Two boxes side by side, as a wrapped-around load is copied: inserted in
// copy order.

// CHECK-LABEL: func.func @two_boxes
// CHECK: %[[E:.*]] = tensor.empty() : tensor<8x32xf32>
// CHECK: %[[A:.*]] = tensor.insert_slice %{{.*}} into %[[E]][0, 0] [8, 20] [1, 1]
// CHECK: tensor.insert_slice %{{.*}} into %[[A]][0, 20] [8, 12] [1, 1]
func.func @two_boxes(%a: memref<8x20xf32>, %b: memref<8x12xf32>) -> tensor<8x32xf32> {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  %d1 = memref.subview %buf[0, 20] [8, 12] [1, 1] : memref<8x32xf32> to memref<8x12xf32, strided<[32, 1], offset: 20>>
  memref.copy %b, %d1 : memref<8x12xf32> to memref<8x12xf32, strided<[32, 1], offset: 20>>
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Left alone: a second reader, a write after the read, a read without
// restrict, a copy that does not always run, a fill after a copy, a
// rank-reducing view.

// CHECK-LABEL: func.func @second_reader
// CHECK: memref.alloc() : memref<8x32xf32>
// CHECK: bufferization.to_tensor %{{.*}} : memref<8x32xf32>
func.func @second_reader(%a: memref<8x20xf32>, %i: index) -> (tensor<8x32xf32>, f32) {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  %v = memref.load %buf[%i, %i] : memref<8x32xf32>
  return %t, %v : tensor<8x32xf32>, f32
}

// CHECK-LABEL: func.func @write_after_read
// CHECK: memref.alloc() : memref<8x32xf32>
// CHECK: bufferization.to_tensor %{{.*}} : memref<8x32xf32>
func.func @write_after_read(%a: memref<8x20xf32>) -> tensor<8x32xf32> {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  return %t : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @not_restrict
// CHECK: memref.alloc() : memref<8x32xf32>
// CHECK: bufferization.to_tensor %{{.*}} : memref<8x32xf32>
func.func @not_restrict(%a: memref<8x20xf32>) -> tensor<8x32xf32> {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  %t = bufferization.to_tensor %buf : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @conditional_copy
// CHECK: memref.alloc() : memref<8x32xf32>
// CHECK: bufferization.to_tensor %{{.*}} : memref<8x32xf32>
func.func @conditional_copy(%a: memref<8x20xf32>, %c: i1) -> tensor<8x32xf32> {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  scf.if %c {
    memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  }
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @fill_after_copy
// CHECK: memref.alloc() : memref<8x32xf32>
// CHECK: bufferization.to_tensor %{{.*}} : memref<8x32xf32>
func.func @fill_after_copy(%a: memref<8x20xf32>) -> tensor<8x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  linalg.fill ins(%cst : f32) outs(%buf : memref<8x32xf32>)
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @rank_reducing_view
// CHECK: memref.alloc() : memref<8x32xf32>
// CHECK: bufferization.to_tensor %{{.*}} : memref<8x32xf32>
func.func @rank_reducing_view(%a: memref<20xf32>) -> tensor<8x32xf32> {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [1, 20] [1, 1] : memref<8x32xf32> to memref<20xf32, strided<[1]>>
  memref.copy %a, %d0 : memref<20xf32> to memref<20xf32, strided<[1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The source is written after the copy and before the read: the box is
// copied where the original copy was, so it holds the earlier values.

// CHECK-LABEL: func.func @source_written_later
// CHECK: %[[BUF:.*]] = memref.alloc() : memref<8x20xf32>
// CHECK: memref.copy %[[A:.*]], %[[BUF]]
// CHECK: linalg.fill {{.*}} outs(%[[A]] : memref<8x20xf32>)
// CHECK: bufferization.to_tensor %[[BUF]] restrict writable
func.func @source_written_later(%a: memref<8x20xf32>) -> tensor<8x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  linalg.fill ins(%cst : f32) outs(%a : memref<8x20xf32>)
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The target is the buffer's alloc itself: it is not raised, and the result
// handle still holds it.

// CHECK-LABEL: func.func @alloc_target
// CHECK: memref.alloc() {kept} : memref<8x32xf32>
func.func @alloc_target(%a: memref<8x20xf32>) -> tensor<8x32xf32> {
  %buf = memref.alloc() : memref<8x32xf32>
  %d0 = memref.subview %buf[0, 0] [8, 20] [1, 1] : memref<8x32xf32> to memref<8x20xf32, strided<[32, 1]>>
  memref.copy %a, %d0 : memref<8x20xf32> to memref<8x20xf32, strided<[32, 1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<8x32xf32> to tensor<8x32xf32>
  return %t : tensor<8x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %a = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %a : (!transform.any_op) -> !transform.any_op
    transform.annotate %r "kept" : !transform.any_op
    transform.yield
  }
}
