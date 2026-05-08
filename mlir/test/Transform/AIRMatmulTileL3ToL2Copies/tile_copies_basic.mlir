//===- tile_copies_basic.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Triton-XDNA-style input: matmul preceded by L3->L2 memref.copy stagings.
// Verifies (1) memref.copy → linalg.copy conversion, (2) per-operand K-tiling,
// (3) loop annotations.

// RUN: air-opt %s '-air-matmul-codegen=bufferize-output-l2=true tile-l3-to-l2-copies=true k-l2-tile=16 do-vec-prep=false' | FileCheck %s

// CHECK-LABEL: func.func @matmul_with_l3_l2_copies
// LHS copy (64x784) is tiled by [0, 16] → outer scf.for over K, copy of 64x16 tiles.
// CHECK:      memref.alloc() : memref<64x784xf32>
// CHECK:      scf.for
// CHECK:        memref.subview {{.*}} [64, 16] [1, 1]
// CHECK:        memref.subview {{.*}} [64, 16] [1, 1]
// CHECK:        linalg.copy ins(%{{.*}} : memref<64x16xf32{{.*}}>) outs(%{{.*}} : memref<64x16xf32{{.*}}>)
// CHECK:      } {copy_a_loop}
// RHS copy (784x32) is tiled by [16, 0] → outer scf.for over K, copy of 16x32 tiles.
// CHECK:      memref.alloc() : memref<784x32xf32>
// CHECK:      scf.for
// CHECK:        memref.subview {{.*}} [16, 32] [1, 1]
// CHECK:        memref.subview {{.*}} [16, 32] [1, 1]
// CHECK:        linalg.copy ins(%{{.*}} : memref<16x32xf32{{.*}}>) outs(%{{.*}} : memref<16x32xf32{{.*}}>)
// CHECK:      } {copy_b_loop}
// CHECK:      linalg.matmul

func.func @matmul_with_l3_l2_copies(%argA: memref<*xf32>, %argB: memref<*xf32>, %argC: memref<*xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %reinterpret_a = memref.reinterpret_cast %argA to offset: [%c0], sizes: [64, 784], strides: [784, 1] : memref<*xf32> to memref<64x784xf32, strided<[784, 1], offset: ?>>
  %alloc_a = memref.alloc() : memref<64x784xf32>
  memref.copy %reinterpret_a, %alloc_a : memref<64x784xf32, strided<[784, 1], offset: ?>> to memref<64x784xf32>
  %ta = bufferization.to_tensor %alloc_a restrict writable : memref<64x784xf32> to tensor<64x784xf32>

  %reinterpret_b = memref.reinterpret_cast %argB to offset: [%c0], sizes: [784, 32], strides: [32, 1] : memref<*xf32> to memref<784x32xf32, strided<[32, 1], offset: ?>>
  %alloc_b = memref.alloc() : memref<784x32xf32>
  memref.copy %reinterpret_b, %alloc_b : memref<784x32xf32, strided<[32, 1], offset: ?>> to memref<784x32xf32>
  %tb = bufferization.to_tensor %alloc_b restrict writable : memref<784x32xf32> to tensor<784x32xf32>

  %tc_init = tensor.empty() : tensor<64x32xf32>
  %tc_fill = linalg.fill ins(%cst : f32) outs(%tc_init : tensor<64x32xf32>) -> tensor<64x32xf32>
  %tc = linalg.matmul ins(%ta, %tb : tensor<64x784xf32>, tensor<784x32xf32>) outs(%tc_fill : tensor<64x32xf32>) -> tensor<64x32xf32>

  %reinterpret_c = memref.reinterpret_cast %argC to offset: [%c0], sizes: [64, 32], strides: [32, 1] : memref<*xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>>
  bufferization.materialize_in_destination %tc in writable %reinterpret_c : (tensor<64x32xf32>, memref<64x32xf32, strided<[32, 1], offset: ?>>) -> ()
  return
}
