//===- air_herd_vectorize.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-herd-vectorize | FileCheck %s

// CHECK-LABEL: func.func @test_herd_vectorize
func.func @test_herd_vectorize() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloc0 = memref.alloc() : memref<32x32xf32, 2 : i32>
  %alloc1 = memref.alloc() : memref<32x32xf32, 2 : i32>
  %alloc2 = memref.alloc() : memref<32x32xf32, 2 : i32>

  // CHECK: air.herd
  air.herd @herd_0 tile (%arg3, %arg4) in (%size_x = %c2, %size_y = %c2) args(%arg5 = %alloc0, %arg6 = %alloc1, %arg7 = %alloc2) : memref<32x32xf32, 2 : i32>, memref<32x32xf32, 2 : i32>, memref<32x32xf32, 2 : i32> {

    // This linalg.generic should be vectorized
    // CHECK: vector.transfer_read
    // CHECK: vector.transfer_read
    // CHECK: arith.addf
    // CHECK: vector.transfer_write
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg5, %arg6 : memref<32x32xf32, 2 : i32>, memref<32x32xf32, 2 : i32>)
      outs(%arg7 : memref<32x32xf32, 2 : i32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }

    air.herd_terminator
  }

  memref.dealloc %alloc0 : memref<32x32xf32, 2 : i32>
  memref.dealloc %alloc1 : memref<32x32xf32, 2 : i32>
  memref.dealloc %alloc2 : memref<32x32xf32, 2 : i32>
  return
}
