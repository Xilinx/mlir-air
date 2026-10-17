//===- linalg_pack_to_dma.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma | FileCheck %s

// CHECK-LABEL: func.func @pack_2d
// CHECK-NOT: linalg.pack
// CHECK: air.dma_memcpy_nd
func.func @pack_2d(%src: memref<256x64xbf16>, %dst: memref<4x1x64x64xbf16, 1>) {
  linalg.pack %src outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %dst : memref<256x64xbf16> -> memref<4x1x64x64xbf16, 1>
  return
}

// CHECK-LABEL: func.func @pack_transposed
// CHECK-NOT: linalg.pack
// CHECK: air.dma_memcpy_nd
func.func @pack_transposed(%src: memref<64x256xbf16>, %dst: memref<4x1x64x64xbf16, 1>) {
  linalg.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %dst : memref<64x256xbf16> -> memref<4x1x64x64xbf16, 1>
  return
}

// CHECK-LABEL: func.func @pack_4d
// CHECK-NOT: linalg.pack
// CHECK: air.dma_memcpy_nd
func.func @pack_4d(%src: memref<1x1x64x64xbf16, 1>, %dst: memref<1x1x8x8x8x8xbf16, 2>) {
  linalg.pack %src outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %dst : memref<1x1x64x64xbf16, 1> -> memref<1x1x8x8x8x8xbf16, 2>
  return
}

// One tile along the packed dim, but outer_dims_perm moves it first: still a
// permutation, not a plain copy.
// CHECK-LABEL: func.func @pack_outer_perm_one_tile
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0, 0, 0, 0] [8, 4, 1, 1, 8] [8, 64, 64, 64, 1])
func.func @pack_outer_perm_one_tile(%src: memref<4x1x1x64xbf16, 1>, %dst: memref<8x4x1x1x8xbf16, 2>) {
  linalg.pack %src outer_dims_perm = [3, 0, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dst : memref<4x1x1x64xbf16, 1> -> memref<8x4x1x1x8xbf16, 2>
  return
}

// CHECK-LABEL: func.func @pack_unit_outer_perm
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0, 0, 0] [2, 8, 1, 8] [8, 16, 8, 1])
func.func @pack_unit_outer_perm(%src: memref<8x2x8xbf16, 1>, %dst: memref<2x8x1x8xbf16, 2>) {
  linalg.pack %src outer_dims_perm = [1, 0, 2] inner_dims_pos = [2] inner_tiles = [8] into %dst : memref<8x2x8xbf16, 1> -> memref<2x8x1x8xbf16, 2>
  return
}

// CHECK-LABEL: func.func @unpack_outer_perm_one_tile
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0, 0, 0, 0] [4, 1, 1, 8, 8] [8, 8, 8, 32, 1])
func.func @unpack_outer_perm_one_tile(%src: memref<8x4x1x1x8xf32, 2>, %dst: memref<4x1x1x64xf32, 1>) {
  linalg.unpack %src outer_dims_perm = [3, 0, 1, 2] inner_dims_pos = [3] inner_tiles = [8] into %dst : memref<8x4x1x1x8xf32, 2> -> memref<4x1x1x64xf32, 1>
  return
}
