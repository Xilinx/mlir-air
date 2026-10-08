//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// A packed buffer P[2][10][4]: rows 0..7 values, row 8 scales, row 9 offsets.
// The three loads become one copy of rows 0..9 and slices of it. The scale
// view's unit row carries an arbitrary stride, which does not matter.
// CHECK-LABEL: @packed_fields
// CHECK: %[[V:.*]] = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [2, 10, 4], strides: [40, 4, 1]
// CHECK: %[[A:.*]] = memref.alloc() : memref<2x10x4xi16, 1>
// CHECK: memref.copy %[[V]], %[[A]]
// CHECK-NOT: memref.copy
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[A]]
// CHECK-DAG: tensor.extract_slice %[[T]][0, 0, 0] [2, 8, 4] [1, 1, 1]
// CHECK-DAG: tensor.extract_slice %[[T]][0, 8, 0] [2, 1, 4] [1, 1, 1]
// CHECK-DAG: tensor.extract_slice %[[T]][0, 9, 0] [2, 1, 4] [1, 1, 1]
func.func @packed_fields(%p: memref<*xi16>, %o: index) -> (tensor<2x8x4xi16>, tensor<2x1x4xi16>, tensor<2x1x4xi16>) {
  %c32 = arith.constant 32 : index
  %c36 = arith.constant 36 : index
  %os = arith.addi %o, %c32 : index
  %om = arith.addi %o, %c36 : index
  %vq = memref.reinterpret_cast %p to offset: [%o], sizes: [2, 8, 4], strides: [40, 4, 1] : memref<*xi16> to memref<2x8x4xi16, strided<[40, 4, 1], offset: ?>>
  %aq = memref.alloc() : memref<2x8x4xi16, 1>
  memref.copy %vq, %aq : memref<2x8x4xi16, strided<[40, 4, 1], offset: ?>> to memref<2x8x4xi16, 1>
  %tq = bufferization.to_tensor %aq restrict writable : memref<2x8x4xi16, 1> to tensor<2x8x4xi16>
  %vs = memref.reinterpret_cast %p to offset: [%os], sizes: [2, 1, 4], strides: [40, 99, 1] : memref<*xi16> to memref<2x1x4xi16, strided<[40, 99, 1], offset: ?>>
  %as = memref.alloc() : memref<2x1x4xi16, 1>
  memref.copy %vs, %as : memref<2x1x4xi16, strided<[40, 99, 1], offset: ?>> to memref<2x1x4xi16, 1>
  %ts = bufferization.to_tensor %as restrict writable : memref<2x1x4xi16, 1> to tensor<2x1x4xi16>
  %vm = memref.reinterpret_cast %p to offset: [%om], sizes: [2, 1, 4], strides: [40, 4, 1] : memref<*xi16> to memref<2x1x4xi16, strided<[40, 4, 1], offset: ?>>
  %am = memref.alloc() : memref<2x1x4xi16, 1>
  memref.copy %vm, %am : memref<2x1x4xi16, strided<[40, 4, 1], offset: ?>> to memref<2x1x4xi16, 1>
  %tm = bufferization.to_tensor %am restrict writable : memref<2x1x4xi16, 1> to tensor<2x1x4xi16>
  return %tq, %ts, %tm : tensor<2x8x4xi16>, tensor<2x1x4xi16>, tensor<2x1x4xi16>
}

// Views of different bases are left alone.
// CHECK-LABEL: @different_bases
// CHECK-COUNT-2: memref.copy
func.func @different_bases(%p: memref<*xi16>, %q: memref<*xi16>, %o: index) -> (tensor<4xi16>, tensor<4xi16>) {
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a0 = memref.alloc() : memref<4xi16>
  memref.copy %v0, %a0 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<4xi16> to tensor<4xi16>
  %v1 = memref.reinterpret_cast %q to offset: [%o], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a1 = memref.alloc() : memref<4xi16>
  memref.copy %v1, %a1 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<4xi16> to tensor<4xi16>
  return %t0, %t1 : tensor<4xi16>, tensor<4xi16>
}

// A write between the copies (here to the packed buffer itself) means the
// later copy reads a different state than the earlier one: no merge.
// CHECK-LABEL: @write_between
// CHECK: memref.copy
// CHECK: memref.store
// CHECK: memref.copy
func.func @write_between(%p: memref<*xi16>, %o: index, %x: memref<64xi16>, %v: i16) -> (tensor<4xi16>, tensor<4xi16>) {
  %c4 = arith.constant 4 : index
  %o1 = arith.addi %o, %c4 : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a0 = memref.alloc() : memref<4xi16, 1>
  memref.copy %v0, %a0 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<4xi16, 1> to tensor<4xi16>
  memref.store %v, %x[%o1] : memref<64xi16>
  %v1 = memref.reinterpret_cast %p to offset: [%o1], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a1 = memref.alloc() : memref<4xi16, 1>
  memref.copy %v1, %a1 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<4xi16, 1> to tensor<4xi16>
  return %t0, %t1 : tensor<4xi16>, tensor<4xi16>
}

// A read of the destination before its copy would see the merged copy before
// it is made: no merge.
// CHECK-LABEL: @read_before_copy
// CHECK-COUNT-2: memref.copy
func.func @read_before_copy(%p: memref<*xi16>, %o: index) -> (tensor<4xi16>, tensor<4xi16>) {
  %c4 = arith.constant 4 : index
  %o1 = arith.addi %o, %c4 : index
  %v0 = memref.reinterpret_cast %p to offset: [%o], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a0 = memref.alloc() : memref<4xi16, 1>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<4xi16, 1> to tensor<4xi16>
  memref.copy %v0, %a0 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16, 1>
  %v1 = memref.reinterpret_cast %p to offset: [%o1], sizes: [4], strides: [1] : memref<*xi16> to memref<4xi16, strided<[1], offset: ?>>
  %a1 = memref.alloc() : memref<4xi16, 1>
  memref.copy %v1, %a1 : memref<4xi16, strided<[1], offset: ?>> to memref<4xi16, 1>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<4xi16, 1> to tensor<4xi16>
  return %t0, %t1 : tensor<4xi16>, tensor<4xi16>
}
