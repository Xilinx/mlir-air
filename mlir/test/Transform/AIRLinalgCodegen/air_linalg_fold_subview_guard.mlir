//===- air_linalg_fold_subview_guard.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-codegen=test-patterns -split-input-file | FileCheck %s

// A subview of a unit-stride subview folds into one subview.
// CHECK-LABEL: @unit_stride_source
// CHECK: %[[S:.*]] = memref.subview %arg0[12, 24] [8, 8] [1, 1]
// CHECK: return %[[S]]
func.func @unit_stride_source(%arg0: memref<64x64xf32>) -> memref<8x8xf32, strided<[64, 1], offset: 792>> {
  %0 = memref.subview %arg0[8, 16] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: 528>>
  %1 = memref.subview %0[4, 8] [8, 8] [1, 1] : memref<32x32xf32, strided<[64, 1], offset: 528>> to memref<8x8xf32, strided<[64, 1], offset: 792>>
  return %1 : memref<8x8xf32, strided<[64, 1], offset: 792>>
}

// -----

// The source stride scales the inner offsets, so this pair is left alone.
// CHECK-LABEL: @strided_source
// CHECK: %[[S:.*]] = memref.subview %arg0[0, 0] [32, 32] [2, 2]
// CHECK: memref.subview %[[S]][4, 8] [8, 8] [1, 1]
func.func @strided_source(%arg0: memref<64x64xf32>) -> memref<8x8xf32, strided<[128, 2], offset: 528>> {
  %0 = memref.subview %arg0[0, 0] [32, 32] [2, 2] : memref<64x64xf32> to memref<32x32xf32, strided<[128, 2]>>
  %1 = memref.subview %0[4, 8] [8, 8] [1, 1] : memref<32x32xf32, strided<[128, 2]>> to memref<8x8xf32, strided<[128, 2], offset: 528>>
  return %1 : memref<8x8xf32, strided<[128, 2], offset: 528>>
}

// -----

// A rank-reducing source has fewer offsets than its own source.
// CHECK-LABEL: @rank_reducing_source
// CHECK: %[[S:.*]] = memref.subview %arg0[3, 0] [1, 64] [1, 1]
// CHECK: memref.subview %[[S]][8] [16] [1]
func.func @rank_reducing_source(%arg0: memref<64x64xf32>) -> memref<16xf32, strided<[1], offset: 200>> {
  %0 = memref.subview %arg0[3, 0] [1, 64] [1, 1] : memref<64x64xf32> to memref<64xf32, strided<[1], offset: 192>>
  %1 = memref.subview %0[8] [16] [1] : memref<64xf32, strided<[1], offset: 192>> to memref<16xf32, strided<[1], offset: 200>>
  return %1 : memref<16xf32, strided<[1], offset: 200>>
}

// -----

// Dynamic offsets on a unit-stride source still fold.
// CHECK-LABEL: @dynamic_offsets
// CHECK-SAME: %[[I:[a-zA-Z0-9]+]]: index, %[[J:[a-zA-Z0-9]+]]: index
// CHECK: %[[O:.*]] = arith.addi %[[I]], %[[J]] : index
// CHECK: %[[S:.*]] = memref.subview %arg0[%[[O]], 16] [8, 8] [1, 1]
// CHECK: return %[[S]]
func.func @dynamic_offsets(%arg0: memref<64x64xf32>, %i: index, %j: index) -> memref<8x8xf32, strided<[64, 1], offset: ?>> {
  %0 = memref.subview %arg0[%i, 16] [32, 32] [1, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
  %1 = memref.subview %0[%j, 0] [8, 8] [1, 1] : memref<32x32xf32, strided<[64, 1], offset: ?>> to memref<8x8xf32, strided<[64, 1], offset: ?>>
  return %1 : memref<8x8xf32, strided<[64, 1], offset: ?>>
}

// -----

// A dynamic source stride is not known to be 1, so nothing folds.
// CHECK-LABEL: @dynamic_stride_source
// CHECK: %[[S:.*]] = memref.subview %arg0[0, 0] [32, 32] [%arg1, 1]
// CHECK: memref.subview %[[S]][4, 8] [8, 8] [1, 1]
func.func @dynamic_stride_source(%arg0: memref<64x64xf32>, %s: index) -> memref<8x8xf32, strided<[?, 1], offset: ?>> {
  %0 = memref.subview %arg0[0, 0] [32, 32] [%s, 1] : memref<64x64xf32> to memref<32x32xf32, strided<[?, 1]>>
  %1 = memref.subview %0[4, 8] [8, 8] [1, 1] : memref<32x32xf32, strided<[?, 1]>> to memref<8x8xf32, strided<[?, 1], offset: ?>>
  return %1 : memref<8x8xf32, strided<[?, 1], offset: ?>>
}
