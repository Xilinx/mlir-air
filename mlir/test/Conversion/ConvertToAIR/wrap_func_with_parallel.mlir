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

// Test scalar SSA chain between linalg ops (issue #1367).
// The pattern: linalg.generic -> linalg.reduce -> tensor.extract ->
// scalar arith chain -> linalg.fill -> linalg.generic should be fully
// moved into the scf.parallel body.

// CHECK-LABEL: @func_scalar_ssa_chain
// CHECK: scf.parallel {{.*}} {
// CHECK:   linalg.generic
// CHECK:   linalg.reduce
// CHECK:   tensor.extract
// CHECK:   arith.divf
// CHECK:   arith.addf
// CHECK:   math.rsqrt
// CHECK:   linalg.fill
// CHECK:   linalg.generic
// CHECK:   scf.reduce

#map_identity = affine_map<(d0) -> (d0)>

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

func.func @func_scalar_ssa_chain(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
  %cst_eps = arith.constant 1.000000e-05 : f32
  %cst_N = arith.constant 6.400000e+01 : f32
  %cst_zero = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.muli %arg6, %c32_i32 : i32
  %1 = arith.index_cast %0 : i32 to index

  // Load input
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [64], strides: [1] : memref<*xbf16> to memref<64xbf16, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<64xbf16>
  memref.copy %reinterpret_cast, %alloc : memref<64xbf16, strided<[1], offset: ?>> to memref<64xbf16>
  %input = bufferization.to_tensor %alloc restrict writable : memref<64xbf16> to tensor<64xbf16>

  // Square: x * x
  %empty_sq = tensor.empty() : tensor<64xbf16>
  %sq = linalg.generic {indexing_maps = [#map_identity, #map_identity], iterator_types = ["parallel"]}
    ins(%input : tensor<64xbf16>) outs(%empty_sq : tensor<64xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %mul = arith.mulf %in, %in : bf16
    linalg.yield %mul : bf16
  } -> tensor<64xbf16>

  // Reduce: sum(x*x)
  %init = tensor.empty() : tensor<f32>
  %fill_init = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<f32>) -> tensor<f32>
  %reduced = linalg.reduce ins(%sq : tensor<64xbf16>) outs(%fill_init : tensor<f32>) dimensions = [0]
    (%in: bf16, %acc: f32) {
      %ext = arith.extf %in : bf16 to f32
      %add = arith.addf %acc, %ext : f32
      linalg.yield %add : f32
    }

  // Scalar SSA chain: extract -> divf -> addf -> rsqrt
  %extracted = tensor.extract %reduced[] : tensor<f32>
  %mean = arith.divf %extracted, %cst_N : f32
  %with_eps = arith.addf %mean, %cst_eps : f32
  %rstd = math.rsqrt %with_eps : f32

  // Broadcast rsqrt and multiply: x * rsqrt
  %empty_out = tensor.empty() : tensor<64xf32>
  %fill_rstd = linalg.fill ins(%rstd : f32) outs(%empty_out : tensor<64xf32>) -> tensor<64xf32>
  %empty_mul = tensor.empty() : tensor<64xf32>
  %result = linalg.generic {indexing_maps = [#map_identity, #map_identity, #map_identity], iterator_types = ["parallel"]}
    ins(%input, %fill_rstd : tensor<64xbf16>, tensor<64xf32>) outs(%empty_mul : tensor<64xf32>) {
  ^bb0(%x: bf16, %r: f32, %out: f32):
    %xf = arith.extf %x : bf16 to f32
    %mul = arith.mulf %xf, %r : f32
    linalg.yield %mul : f32
  } -> tensor<64xf32>

  // Store output
  %reinterpret_cast_out = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [64], strides: [1] : memref<*xbf16> to memref<64xbf16, strided<[1], offset: ?>>
  %empty_trunc = tensor.empty() : tensor<64xbf16>
  %truncated = linalg.generic {indexing_maps = [#map_identity, #map_identity], iterator_types = ["parallel"]}
    ins(%result : tensor<64xf32>) outs(%empty_trunc : tensor<64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %tr = arith.truncf %in : f32 to bf16
    linalg.yield %tr : bf16
  } -> tensor<64xbf16>
  bufferization.materialize_in_destination %truncated in writable %reinterpret_cast_out : (tensor<64xbf16>, memref<64xbf16, strided<[1], offset: ?>>) -> ()
  return
}
