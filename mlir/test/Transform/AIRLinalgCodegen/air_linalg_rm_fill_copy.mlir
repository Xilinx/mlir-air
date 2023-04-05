//===- air_linalg_rm_fill_copy.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-codegen=test-patterns | FileCheck %s

// CHECK-LABEL: test_0
// CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[A0:.*]] = memref.alloc() : memref<64x64xf32>
// CHECK-NEXT: linalg.fill ins(%[[C0]] : f32) outs(%[[A0]] : memref<64x64xf32>
// CHECK-NEXT: linalg.matmul {{.*}} outs(%[[A0]] : memref<64x64xf32>
// CHECK-NEXT: %[[A1:.*]] = memref.alloc()
// CHECK-NEXT: linalg.fill ins(%[[C0]] : f32) outs(%[[A1]] : memref<64x64xf32>
// CHECK-NEXT: linalg.matmul {{.*}} outs(%[[A1]] : memref<64x64xf32>
// CHECK-NEXT: %[[A2:.*]] = memref.alloc() : memref<64x64xf32>
// CHECK-NEXT: linalg.fill ins(%[[C0]] : f32) outs(%[[A2]] : memref<64x64xf32>
// CHECK-NEXT: linalg.matmul {{.*}} outs(%[[A2]] : memref<64x64xf32>
func.func @test_0(%arg0: memref<64x256xf32>, %arg1: memref<256x64xf32>, %arg2: memref<256x64xf32>, %arg3: memref<256x64xf32>) -> memref<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<64x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
  %alloc_0 = memref.alloc() : memref<64x64xf32>
  memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
  linalg.matmul ins(%arg0, %arg1 : memref<64x256xf32>, memref<256x64xf32>) outs(%alloc_0 : memref<64x64xf32>)
  %alloc_1 = memref.alloc() : memref<64x64xf32>
  memref.copy %alloc, %alloc_1 : memref<64x64xf32> to memref<64x64xf32>
  linalg.matmul ins(%arg0, %arg2 : memref<64x256xf32>, memref<256x64xf32>) outs(%alloc_1 : memref<64x64xf32>)
  %alloc_2 = memref.alloc() : memref<64x64xf32>
  memref.copy %alloc, %alloc_2 : memref<64x64xf32> to memref<64x64xf32>
  linalg.matmul ins(%alloc_0, %alloc_1 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc_2 : memref<64x64xf32>)
  return %alloc_2 : memref<64x64xf32>
}