//===- wrap_func_with_parallel.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-wrap-func-with-parallel='loop-bounds=4,4,1' %s | FileCheck %s
// CHECK-LABEL: @func0
// CHECK: scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]]) = (%c0{{.*}}, %c0{{.*}}, %c0{{.*}}) to (%c4{{.*}}, %c4{{.*}}, %c1{{.*}}) step (%c1{{.*}}, %c1{{.*}}, %c1{{.*}})
// CHECK: arith.index_cast %[[ARG0]] : index to i32
// CHECK: arith.index_cast %[[ARG1]] : index to i32

func.func @func0(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.muli %arg6, %c32_i32 : i32
  %1 = arith.index_cast %0 : i32 to index
  %2 = arith.muli %arg7, %c32_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %4 = arith.muli %1, %c64 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [32, 64], strides: [%c64, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
  %alloc = memref.alloc() : memref<32x64xf32>
  memref.copy %reinterpret_cast, %alloc : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
  %5 = bufferization.to_tensor %alloc restrict writable : memref<32x64xf32> to tensor<32x64xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [64, 32], strides: [%c64, 1] : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
  %alloc_1 = memref.alloc() : memref<64x32xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
  %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x32xf32> to tensor<64x32xf32>
  %7 = tensor.empty() : tensor<32x32xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %9 = linalg.matmul ins(%5, %6 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %10 = arith.addi %4, %3 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [%c64, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
  bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<32x32xf32>, memref<32x32xf32, strided<[?, 1], offset: ?>>) -> ()
  return
}
