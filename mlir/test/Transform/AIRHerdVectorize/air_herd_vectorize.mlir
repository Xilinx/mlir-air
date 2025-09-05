//===- air_herd_vectorize.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-herd-vectorize | FileCheck %s

// CHECK-LABEL: func.func @test_herd_vectorize
func.func @test_herd_vectorize(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // CHECK: air.herd
  air.herd @herd_0 tile (%arg3, %arg4) in (%size_x = %c2, %size_y = %c2) args(%arg5 = %arg0, %arg6 = %arg1, %arg7 = %arg2) : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    
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
    } ins(%arg5, %arg6 : memref<32x32xf32>, memref<32x32xf32>) 
      outs(%arg7 : memref<32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    
    air.herd_terminator
  }
  
  return
}
