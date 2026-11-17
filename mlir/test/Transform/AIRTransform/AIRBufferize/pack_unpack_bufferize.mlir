//===- pack_unpack_bufferize.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -one-shot-bufferize=bufferize-function-boundaries | FileCheck %s

// linalg.pack and linalg.unpack bufferize in place; air-copy-to-dma lowers
// them to DMAs later.

// CHECK-LABEL: func.func @pack
// CHECK: linalg.pack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %{{.*}} : memref<64x32xbf16
func.func @pack(%src: tensor<64x32xbf16>, %dst: tensor<4x8x8x8xbf16>) -> tensor<4x8x8x8xbf16> {
  %0 = linalg.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dst : tensor<64x32xbf16> -> tensor<4x8x8x8xbf16>
  return %0 : tensor<4x8x8x8xbf16>
}

// CHECK-LABEL: func.func @unpack
// CHECK: linalg.unpack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %{{.*}} : memref<4x8x8x8xbf16
func.func @unpack(%src: tensor<4x8x8x8xbf16>, %dst: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
  %0 = linalg.unpack %src outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dst : tensor<4x8x8x8xbf16> -> tensor<64x32xbf16>
  return %0 : tensor<64x32xbf16>
}
