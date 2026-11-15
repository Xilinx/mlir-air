//===- opt_shim_dma_bds_cast_offset.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// An offset read from an i32 argument, in a dimension that is folded into the
// next one: the folded offset applies to the cast's index result.

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 512)>
// CHECK-LABEL: func.func @cast_offset
// CHECK: %[[I:.*]] = arith.index_cast %arg1 : i32 to index
// CHECK: %[[O:.*]] = affine.apply #[[MAP]]()[%[[I]]]
// CHECK: air.channel.get {{.*}}(%arg0[%[[O]]] [%c4096{{.*}}] [%c1{{.*}}])
func.func @cast_offset(%arg0: memref<2x256x512xbf16>, %arg1: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c512 = arith.constant 512 : index
  %c131072 = arith.constant 131072 : index
  %i = arith.index_cast %arg1 : i32 to index
  %0 = air.channel.get async @channel_0[] (%arg0[%c0, %i, %c0] [%c1, %c8, %c512] [%c131072, %c512, %c1]) {metadata = @airMemcpyId1} : (memref<2x256x512xbf16>)
  return
}

// -----

// The next dimension's own offset is added to the folded one.

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 512 + 16)>
// CHECK-LABEL: func.func @cast_offset_next_offset
// CHECK: %[[I:.*]] = arith.index_cast %arg1 : i32 to index
// CHECK: %[[O:.*]] = affine.apply #[[MAP]]()[%[[I]]]
// CHECK: air.channel.get {{.*}}(%arg0[%[[O]]] [%c4096{{.*}}] [%c1{{.*}}])
func.func @cast_offset_next_offset(%arg0: memref<2x256x512xbf16>, %arg1: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  %c131072 = arith.constant 131072 : index
  %i = arith.index_cast %arg1 : i32 to index
  %0 = air.channel.get async @channel_0[] (%arg0[%c0, %i, %c16] [%c1, %c8, %c512] [%c131072, %c512, %c1]) {metadata = @airMemcpyId1} : (memref<2x256x512xbf16>)
  return
}

// -----

// The cast offset in the outermost dimension, folded across both inner ones.

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 131072)>
// CHECK-LABEL: func.func @cast_offset_outer
// CHECK: %[[I:.*]] = arith.index_cast %arg1 : i32 to index
// CHECK: %[[O:.*]] = affine.apply #[[MAP]]()[%[[I]]]
// CHECK: air.channel.get {{.*}}(%arg0[%[[O]]] [%c131072{{.*}}] [%c1{{.*}}])
func.func @cast_offset_outer(%arg0: memref<4x256x512xbf16>, %arg1: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c131072 = arith.constant 131072 : index
  %i = arith.index_cast %arg1 : i32 to index
  %0 = air.channel.get async @channel_0[] (%arg0[%i, %c0, %c0] [%c1, %c256, %c512] [%c131072, %c512, %c1]) {metadata = @airMemcpyId1} : (memref<4x256x512xbf16>)
  return
}
