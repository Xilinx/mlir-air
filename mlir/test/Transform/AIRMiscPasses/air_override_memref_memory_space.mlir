//===- air_override_memref_memory_space.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-override-memref-memory-space="scope=herd memory-space=2" | FileCheck %s
// RUN: air-opt %s -air-override-memref-memory-space="scope=launch memory-space=2" | FileCheck %s --check-prefix=LAUNCH

module {

  // CHECK-LABEL: func.func @func0
  // CHECK: memref.alloc() : memref<32x64xf32, 2 : i32>
  // LAUNCH-LABEL: func.func @func0
  // LAUNCH: memref.alloc() : memref<32x64xf32, 2 : i32>
  // MS1-LABEL: func.func @func0
  // MS1: memref.alloc() : memref<32x64xf32, 2 : i32>

  func.func @func0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg9, %arg10) in (%arg11=%c1, %arg12=%c1_0) args(%arg13=%arg0, %arg14=%arg1, %arg15=%arg2) : memref<*xf32>, memref<*xf32>, memref<*xf32> {
      air.segment @bare_matmul_0  args(%arg16=%arg9, %arg17=%arg10, %arg18=%arg11, %arg19=%arg12, %arg20=%arg13, %arg21=%arg14, %arg22=%arg15) : index, index, index, index, memref<*xf32>, memref<*xf32>, memref<*xf32> {
        %c2_1 = arith.constant 2 : index
        %c2_2 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg23, %arg24) in (%arg25=%c2_1, %arg26=%c2_2) args(%arg27=%arg16, %arg28=%arg17, %arg29=%arg18, %arg30=%arg19, %arg31=%arg20, %arg32=%arg21, %arg33=%arg22) : index, index, index, index, memref<*xf32>, memref<*xf32>, memref<*xf32> {
          %c32 = arith.constant 32 : index
          %c2048 = arith.constant 2048 : index
          %c64 = arith.constant 64 : index
          %1 = arith.muli %arg23, %c2048 : index
          %reinterpret_cast = memref.reinterpret_cast %arg31 to offset: [%1], sizes: [32, 64], strides: [%c64, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<32x64xf32, 3>
          memref.copy %reinterpret_cast, %alloc : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32, 3>
        }
      }
    }
    return
  }

  // LAUNCH-LABEL: func.func @func1
  // LAUNCH: memref.alloc() : memref<8x4x4x8xf32, 2 : i32>
  // LAUNCH: memref.collapse_shape {{.*}} : memref<8x4x4x8xf32, 2 : i32> into memref<32x32xf32, 2 : i32>
  // LAUNCH: memref.alloc() : memref<4x8x8x4xf32, 2 : i32>
  // LAUNCH: memref.collapse_shape {{.*}} : memref<4x8x8x4xf32, 2 : i32> into memref<32x32xf32, 2 : i32>
  // LAUNCH: memref.alloc() : memref<32x32xf32, 2 : i32>
  // LAUNCH: memref.expand_shape {{.*}} : memref<32x32xf32, 2 : i32> into memref<8x4x8x4xf32, 2 : i32>
  // LAUNCH: memref.alloc() : memref<8x8x4x4xf32, 2 : i32>
  // MS1-LABEL: func.func @func1
  // MS1: memref.alloc() : memref<8x4x4x8xf32, 1 : i32>
  // MS1: memref.collapse_shape {{.*}} : memref<8x4x4x8xf32, 1 : i32> into memref<32x32xf32, 1 : i32>
  // MS1: memref.alloc() : memref<4x8x8x4xf32, 1 : i32>
  // MS1: memref.collapse_shape {{.*}} : memref<4x8x8x4xf32, 1 : i32> into memref<32x32xf32, 1 : i32>
  // MS1: memref.alloc() : memref<32x32xf32, 1 : i32>
  // MS1: memref.expand_shape {{.*}} : memref<32x32xf32, 1 : i32> into memref<8x4x8x4xf32, 1 : i32>
  // MS1: memref.alloc() : memref<8x8x4x4xf32, 1 : i32>

  func.func @func1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c2, %arg13=%c2, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xf32>, memref<*xf32>, memref<*xf32> {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<8x4x4x8xf32>
      %collapse_shape = memref.collapse_shape %alloc_2 [[0, 1], [2, 3]] : memref<8x4x4x8xf32> into memref<32x32xf32>
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x8x8x4xf32>
      %collapse_shape_4 = memref.collapse_shape %alloc_3 [[0, 1], [2, 3]] : memref<4x8x8x4xf32> into memref<32x32xf32>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      linalg.matmul ins(%collapse_shape, %collapse_shape_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
      %expand_shape = memref.expand_shape %alloc_5 [[0, 1], [2, 3]] output_shape [8, 4, 8, 4] : memref<32x32xf32> into memref<8x4x8x4xf32>
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8x8x4x4xf32>
      linalg.transpose ins(%expand_shape : memref<8x4x8x4xf32>) outs(%alloc_6 : memref<8x8x4x4xf32>) permutation = [0, 2, 1, 3] 
    }
    return
  }
}
