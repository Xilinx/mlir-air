//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.addf
// CHECK: vector.transfer_write

module {
  func.func @test_herd_vectorize(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.herd @herd_0 tile (%tx, %ty) in (%size_x = %c2, %size_y = %c2) args(%a0=%arg0, %a1=%arg1, %a2=%arg2) : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %subview_a0 = memref.subview %a0[%tx, %ty] [16, 16] [1, 1] : memref<32x32xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
      %subview_a1 = memref.subview %a1[%tx, %ty] [16, 16] [1, 1] : memref<32x32xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
      %subview_a2 = memref.subview %a2[%tx, %ty] [16, 16] [1, 1] : memref<32x32xf32> to memref<16x16xf32, strided<[32, 1], offset: ?>>
      
      // This linalg.add should be vectorized by our transform op
      linalg.add ins(%subview_a0, %subview_a1 : memref<16x16xf32, strided<[32, 1], offset: ?>>, memref<16x16xf32, strided<[32, 1], offset: ?>>) 
                 outs(%subview_a2 : memref<16x16xf32, strided<[32, 1], offset: ?>>)
      
      air.herd_terminator
    }
    return
  }
}
