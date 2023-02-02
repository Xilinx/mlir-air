//===- transform-ops.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: linalg_promote_L1
// CHECK: memref.copy {{.*}} to memref<1024x64xf32, 2>
// CHECK-NEXT: memref.copy {{.*}} to memref<64x1024xf32, 2>
// CHECK-NEXT: linalg.matmul
func.func @linalg_promote_L1(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[0, %c0] [1024, 64] [1, 1] : memref<1024x1024xf32> to memref<1024x64xf32, strided<[1024, 1], offset: ?>>
  %subview_0 = memref.subview %arg1[%c0, 0] [64, 1024] [1, 1] : memref<1024x1024xf32> to memref<64x1024xf32, strided<[1024, 1], offset: ?>>
  linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%subview, %subview_0 : memref<1024x64xf32, strided<[1024, 1], offset: ?>>, memref<64x1024xf32, strided<[1024, 1], offset: ?>>) outs(%arg2 : memref<1024x1024xf32>)
  return
}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = transform.air.linalg_promote %0
  }
}

// -----

// CHECK-LABEL: linalg_promote_L2
// CHECK: memref.copy {{.*}} to memref<1024x64xf32, 1>
// CHECK-NEXT: memref.copy {{.*}} to memref<64x1024xf32, 1>
// CHECK-NEXT: linalg.matmul
func.func @linalg_promote_L2(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[0, %c0] [1024, 64] [1, 1] : memref<1024x1024xf32> to memref<1024x64xf32, strided<[1024, 1], offset: ?>>
  %subview_0 = memref.subview %arg1[%c0, 0] [64, 1024] [1, 1] : memref<1024x1024xf32> to memref<64x1024xf32, strided<[1024, 1], offset: ?>>
  linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%subview, %subview_0 : memref<1024x64xf32, strided<[1024, 1], offset: ?>>, memref<64x1024xf32, strided<[1024, 1], offset: ?>>) outs(%arg2 : memref<1024x1024xf32>)
  return
}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = transform.air.linalg_promote %0 {memory_space="L2"}
  }
}

// -----

// CHECK-LABEL: linalg_promote_one
// CHECK: %[[SV0:.*]] = memref.subview
// CHECK: %[[A0:.*]] = memref.alloc
// CHECK: memref.copy {{.*}} to memref<1024x64xf32, 2>
// CHECK: linalg.matmul {{.*}} ins(%[[A0]], %[[SV0]]
func.func @linalg_promote_one(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[0, %c0] [1024, 64] [1, 1] : memref<1024x1024xf32> to memref<1024x64xf32, strided<[1024, 1], offset: ?>>
  %subview_0 = memref.subview %arg1[%c0, 0] [64, 1024] [1, 1] : memref<1024x1024xf32> to memref<64x1024xf32, strided<[1024, 1], offset: ?>>
  linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%subview, %subview_0 : memref<1024x64xf32, strided<[1024, 1], offset: ?>>, memref<64x1024xf32, strided<[1024, 1], offset: ?>>) outs(%arg2 : memref<1024x1024xf32>)
  return
}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = transform.air.linalg_promote %0 {operands_to_promote=[0]}
  }
}

// -----

func.func @matmul_on_buffers(%arg0: memref<128x128xi32>, %arg1: memref<128x128xi32>, %arg2: memref<128x128xi32>) {
  linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : memref<128x128xi32>, memref<128x128xi32>) outs(%arg2 : memref<128x128xi32>)
  return
}