//===- air_force_l1_memref_in_herd.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-force-l1-memref-in-herd | FileCheck %s

// CHECK-LABEL: func.func @func0
// CHECK: memref.alloc() : memref<32x64xf32, 2>
// CHECK: memref.alloc() : memref<64x32xf32, 2>
// CHECK: memref.alloc() : memref<32x32xf32, 2>

module {
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
          %cst = arith.constant 0.000000e+00 : f32
          %0 = arith.muli %arg24, %c32 : index
          %1 = arith.muli %arg23, %c2048 : index
          %reinterpret_cast = memref.reinterpret_cast %arg31 to offset: [%1], sizes: [32, 64], strides: [%c64, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<32x64xf32>
          memref.copy %reinterpret_cast, %alloc : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg32 to offset: [%0], sizes: [64, 32], strides: [%c64, 1] : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
          %alloc_4 = memref.alloc() : memref<64x32xf32>
          memref.copy %reinterpret_cast_3, %alloc_4 : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_5 : memref<32x32xf32>)
          linalg.matmul ins(%alloc, %alloc_4 : memref<32x64xf32>, memref<64x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
          %2 = arith.addi %1, %0 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg33 to offset: [%2], sizes: [32, 32], strides: [%c64, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
          memref.copy %alloc_5, %reinterpret_cast_6 : memref<32x32xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
        }
      }
    }
    return
  }
}
