//===- forward_copy_into_destination.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -split-input-file %s | FileCheck %s

// A load stored unchanged, as triton-shared lowers it: one copy from the
// source into the destination, and no buffer. The source and the destination
// must be provably disjoint (here the source is `llvm.noalias`), since the
// buffer kept the source as it was and a memref.copy between overlapping
// memory is undefined.

// CHECK-LABEL: func.func @forward
// CHECK-NOT: memref.alloc
// CHECK: %[[D:.*]] = memref.subview %{{.*}}[0] [256] [1]
// CHECK-NEXT: memref.copy %{{.*}}, %[[D]] : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16, strided<[1]>>
// CHECK-NOT: bufferization
// CHECK: return
func.func @forward(%src: memref<256xbf16, strided<[1], offset: ?>> {llvm.noalias}, %home: memref<1024xbf16>) {
  %buf = memref.alloc() : memref<256xbf16>
  memref.copy %src, %buf : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16>
  %t = bufferization.to_tensor %buf restrict writable : memref<256xbf16> to tensor<256xbf16>
  %d = memref.subview %home[0] [256] [1] : memref<1024xbf16> to memref<256xbf16, strided<[1]>>
  bufferization.materialize_in_destination %t in writable %d : (tensor<256xbf16>, memref<256xbf16, strided<[1]>>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The buffer's dealloc goes with it.

// CHECK-LABEL: func.func @dealloc
// CHECK-NOT: memref.alloc
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<64xf32> to memref<64xf32>
// CHECK-NOT: memref.dealloc
func.func @dealloc(%src: memref<64xf32> {llvm.noalias}, %dst: memref<64xf32>) {
  %buf = memref.alloc() : memref<64xf32>
  memref.copy %src, %buf : memref<64xf32> to memref<64xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  memref.dealloc %buf : memref<64xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The source written between the copy and the store: the buffer holds the
// old data, so it stays.

// CHECK-LABEL: func.func @written_between
// CHECK: %[[BUF:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[BUF]]
// CHECK: memref.store
// CHECK: bufferization.materialize_in_destination
func.func @written_between(%src: memref<64xf32> {llvm.noalias}, %dst: memref<64xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %buf = memref.alloc() : memref<64xf32>
  memref.copy %src, %buf : memref<64xf32> to memref<64xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  memref.store %v, %src[%c0] : memref<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The loaded tensor is also computed with: the buffer stays.

// CHECK-LABEL: func.func @also_read
// CHECK: %[[BUF:.*]] = memref.alloc()
// CHECK: memref.copy %{{.*}}, %[[BUF]]
// CHECK: bufferization.materialize_in_destination
func.func @also_read(%src: memref<64xf32> {llvm.noalias}, %dst: memref<64xf32>) -> tensor<64xf32> {
  %buf = memref.alloc() : memref<64xf32>
  memref.copy %src, %buf : memref<64xf32> to memref<64xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return %t : tensor<64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A buffer two copies fill: not one source to forward, so it stays.

// CHECK-LABEL: func.func @two_copies
// CHECK: %[[BUF:.*]] = memref.alloc()
// CHECK-COUNT-2: memref.copy
// CHECK: bufferization.materialize_in_destination
func.func @two_copies(%a: memref<32xf32> {llvm.noalias}, %b: memref<32xf32> {llvm.noalias}, %dst: memref<64xf32>) {
  %buf = memref.alloc() : memref<64xf32>
  %lo = memref.subview %buf[0] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1]>>
  %hi = memref.subview %buf[32] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1], offset: 32>>
  memref.copy %a, %lo : memref<32xf32> to memref<32xf32, strided<[1]>>
  memref.copy %b, %hi : memref<32xf32> to memref<32xf32, strided<[1], offset: 32>>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A masked load stored unchanged under the same mask, as triton-shared
// lowers it: raised, its box read back out, then forwarded to one copy of
// the box.

// CHECK-LABEL: func.func @masked
// CHECK-NOT: memref.alloc
// CHECK-NOT: linalg.fill
// CHECK: %[[S:.*]] = memref.subview %{{.*}}[0] [40] [1]
// CHECK: %[[D:.*]] = memref.subview %{{.*}}[0] [40] [1]
// CHECK: memref.copy %[[S]], %[[D]]
// CHECK-NOT: bufferization
func.func @masked(%src: memref<64xf32, strided<[1], offset: ?>> {llvm.noalias}, %home: memref<64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %buf = memref.alloc() : memref<64xf32>
  linalg.fill ins(%cst : f32) outs(%buf : memref<64xf32>)
  %s = memref.subview %src[0] [40] [1] : memref<64xf32, strided<[1], offset: ?>> to memref<40xf32, strided<[1], offset: ?>>
  %b = memref.subview %buf[0] [40] [1] : memref<64xf32> to memref<40xf32, strided<[1]>>
  memref.copy %s, %b : memref<40xf32, strided<[1], offset: ?>> to memref<40xf32, strided<[1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  %e = tensor.extract_slice %t[0] [40] [1] : tensor<64xf32> to tensor<40xf32>
  %d = memref.subview %home[0] [40] [1] : memref<64xf32> to memref<40xf32, strided<[1]>>
  bufferization.materialize_in_destination %e in writable %d : (tensor<40xf32>, memref<40xf32, strided<[1]>>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.raise_load_buffer_to_tensor %f : (!transform.any_op) -> !transform.any_op
    %s = transform.air.fold_extract_slice_of_insert_slice %r : (!transform.any_op) -> !transform.any_op
    %c = transform.air.forward_copy_into_destination %s : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Source and destination may overlap (function arguments not marked
// `llvm.noalias`): the buffer stays.

// CHECK-LABEL: func.func @may_overlap
// CHECK: memref.alloc()
// CHECK: bufferization.materialize_in_destination
func.func @may_overlap(%src: memref<64xf32>, %dst: memref<64xf32>) {
  %buf = memref.alloc() : memref<64xf32>
  memref.copy %src, %buf : memref<64xf32> to memref<64xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Two views of one buffer, [0, 4) into [1, 5): the buffer keeps the source as
// it was, so it stays.

// CHECK-LABEL: func.func @same_buffer
// CHECK: memref.alloc()
// CHECK: bufferization.materialize_in_destination
func.func @same_buffer(%m: memref<8xf32> {llvm.noalias}) {
  %src = memref.subview %m[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %dst = memref.subview %m[1] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 1>>
  %buf = memref.alloc() : memref<4xf32>
  memref.copy %src, %buf : memref<4xf32, strided<[1]>> to memref<4xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<4xf32> to tensor<4xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<4xf32>, memref<4xf32, strided<[1], offset: 1>>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Two views of one buffer whose elements do not meet, [16, 32) into [32, 48),
// the second through a memref.cast that drops its static offset: forwarded.

// CHECK-LABEL: func.func @same_buffer_apart
// CHECK-NOT: memref.alloc
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<16xf32, strided<[1], offset: 16>> to memref<16xf32, strided<[1], offset: ?>>
// CHECK-NOT: bufferization
func.func @same_buffer_apart(%m: memref<64xf32>) {
  %src = memref.subview %m[16] [16] [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: 16>>
  %view = memref.subview %m[32] [16] [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: 32>>
  %dst = memref.cast %view : memref<16xf32, strided<[1], offset: 32>> to memref<16xf32, strided<[1], offset: ?>>
  %buf = memref.alloc() : memref<16xf32>
  memref.copy %src, %buf : memref<16xf32, strided<[1], offset: 16>> to memref<16xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<16xf32> to tensor<16xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<16xf32>, memref<16xf32, strided<[1], offset: ?>>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The source written in between by a store nested in an op that declares no
// write itself (a tensor-form linalg.generic): the buffer stays.

// CHECK-LABEL: func.func @nested_write
// CHECK: memref.alloc()
// CHECK: linalg.generic
// CHECK: bufferization.materialize_in_destination
#id = affine_map<(d0) -> (d0)>
func.func @nested_write(%src: memref<64xf32> {llvm.noalias}, %dst: memref<64xf32>, %x: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %buf = memref.alloc() : memref<64xf32>
  memref.copy %src, %buf : memref<64xf32> to memref<64xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%x : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    memref.store %in, %src[%c0] : memref<64xf32>
    linalg.yield %in : f32
  } -> tensor<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return %y : tensor<64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A materialization given as the target itself stays, so the returned handle
// still holds it.

// CHECK-LABEL: func.func @target_itself
// CHECK: memref.alloc()
// CHECK: bufferization.materialize_in_destination
func.func @target_itself(%src: memref<64xf32> {llvm.noalias}, %dst: memref<64xf32>) {
  %buf = memref.alloc() : memref<64xf32>
  memref.copy %src, %buf : memref<64xf32> to memref<64xf32>
  %t = bufferization.to_tensor %buf restrict writable : memref<64xf32> to tensor<64xf32>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %m = transform.structured.match ops{["bufferization.materialize_in_destination"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.forward_copy_into_destination %m : (!transform.any_op) -> !transform.any_op
    %p = transform.get_parent_op %r : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
