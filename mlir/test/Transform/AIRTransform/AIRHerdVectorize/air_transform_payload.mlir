//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s
// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DEPRECATED

// transform.structured.vectorize_children_and_apply_patterns on the herd gives
// the same IR.
// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s -o %t.air
// RUN: air-opt -air-transform='filename=%S/upstream_transform.mlir' %s -o %t.upstream
// RUN: diff %t.air %t.upstream

// DEPRECATED: warning: transform.air.herd_vectorize is deprecated

// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.addf
// CHECK: vector.transfer_write

module {
  func.func @test_herd_vectorize() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc0 = memref.alloc() : memref<32x32xf32, 2 : i32>
    %alloc1 = memref.alloc() : memref<32x32xf32, 2 : i32>
    %alloc2 = memref.alloc() : memref<32x32xf32, 2 : i32>
    air.herd @herd_0 tile (%tx, %ty) in (%size_x = %c2, %size_y = %c2) args(%a0=%alloc0, %a1=%alloc1, %a2=%alloc2) : memref<32x32xf32, 2 : i32>, memref<32x32xf32, 2 : i32>, memref<32x32xf32, 2 : i32> {
      %subview_a0 = memref.subview %a0[%tx, %ty] [16, 16] [1, 1] : memref<32x32xf32, 2 : i32> to memref<16x16xf32, strided<[32, 1], offset: ?>, 2 : i32>
      %subview_a1 = memref.subview %a1[%tx, %ty] [16, 16] [1, 1] : memref<32x32xf32, 2 : i32> to memref<16x16xf32, strided<[32, 1], offset: ?>, 2 : i32>
      %subview_a2 = memref.subview %a2[%tx, %ty] [16, 16] [1, 1] : memref<32x32xf32, 2 : i32> to memref<16x16xf32, strided<[32, 1], offset: ?>, 2 : i32>

      // This linalg.elementwise <add> should be vectorized by our transform op
      linalg.elementwise <add> ins(%subview_a0, %subview_a1 : memref<16x16xf32, strided<[32, 1], offset: ?>, 2 : i32>, memref<16x16xf32, strided<[32, 1], offset: ?>, 2 : i32>)
                 outs(%subview_a2 : memref<16x16xf32, strided<[32, 1], offset: ?>, 2 : i32>)

      air.herd_terminator
    }
    memref.dealloc %alloc0 : memref<32x32xf32, 2 : i32>
    memref.dealloc %alloc1 : memref<32x32xf32, 2 : i32>
    memref.dealloc %alloc2 : memref<32x32xf32, 2 : i32>
    return
  }
}
