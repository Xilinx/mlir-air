//===- transform-ops.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// TODO: remove test-transform-dialect-interpreter
// XFAIL: *

// CHECK-LABEL: linalg_promote_L1
// CHECK: memref.copy {{.*}} to memref<{{.*}}, 2>
// CHECK-NEXT: memref.copy {{.*}} to memref<{{.*}}, 2>
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
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = transform.air.linalg_promote %0
  }
}

// -----

// CHECK-LABEL: linalg_promote_L2
// CHECK: memref.copy {{.*}} to memref<{{.*}}, 1>
// CHECK-NEXT: memref.copy {{.*}} to memref<{{.*}}, 1>
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
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = transform.air.linalg_promote %0 {memory_space="L2", operands_to_promote=[0,1,2]}
  }
}

// -----

// CHECK-LABEL: linalg_promote_one
// CHECK: %[[SV0:.*]] = memref.subview
// CHECK: %[[A0:.*]] = memref.alloc
// CHECK: memref.copy {{.*}} to memref<{{.*}}, 2>
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
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = transform.air.linalg_promote %0 {operands_to_promote=[0]}
  }
}

// -----

// CHECK-LABEL: linalg_promote_multi_op
// CHECK: %[[A0:.*]] = memref.alloc() : memref<16x16xf32, 2>
// CHECK-NEXT: linalg.fill ins(%{{.*}} : f32) outs(%[[A0]] : memref<16x16xf32, 2>)
// CHECK-NEXT: linalg.matmul {{.*}} outs(%[[A0]] : memref<16x16xf32, 2>)
// CHECK-NEXT: memref.copy %[[A0]], %{{.*}} : memref<16x16xf32, 2> to memref<16x16xf32, strided<[1024, 1]>>
func.func @linalg_promote_multi_op(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
  %subview0 = memref.subview %arg0[0, 0] [16, 16] [1, 1] : memref<1024x1024xf32> to memref<16x16xf32, strided<[1024, 1]>>
  %subview1 = memref.subview %arg1[0, 0] [16, 16] [1, 1] : memref<1024x1024xf32> to memref<16x16xf32, strided<[1024, 1]>>
  %subview2 = memref.subview %arg2[0, 0] [16, 16] [1, 1] : memref<1024x1024xf32> to memref<16x16xf32, strided<[1024, 1]>>
  %c0 = arith.constant 0.0 : f32
  linalg.fill ins(%c0 : f32) outs(%subview2 : memref<16x16xf32, strided<[1024, 1]>>)
  linalg.matmul ins(%subview0, %subview1 : memref<16x16xf32, strided<[1024, 1]>>, memref<16x16xf32, strided<[1024, 1]>>) outs(%subview2 : memref<16x16xf32, strided<[1024, 1]>>)
  return
}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %2 = transform.merge_handles %0, %1 : !pdl.operation
    transform.air.linalg_promote %2 {"group_size"=2, "operands_to_promote"=[1,4], "memory_space"="L1"}
  }
}
