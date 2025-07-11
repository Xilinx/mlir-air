//===- air_linalg_resolve_tensor_conflicts.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-resolve-tensor-opoperand-conflicts -split-input-file | FileCheck %s

// Element-wise op, whose output OpOperand uses the same tensor as one of the input OpOperands.
// CHECK-LABEL: @func0
// CHECK:  %[[VAL_1:.*]] = bufferization.to_tensor
// CHECK:  %[[VAL_2:.*]] = bufferization.to_tensor
// CHECK:  %[[VAL_3:.*]] = bufferization.alloc_tensor
// CHECK:  %[[VAL_4:.*]] = linalg.generic{{.*}}ins(%[[VAL_1]], %[[VAL_2]] : tensor<8x1xf32>, tensor<8x1xf32>) outs(%[[VAL_3]] : tensor<8x1xf32>)

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @func0(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32) {
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.muli %arg3, %c8_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [8, 1], strides: [1, 1] : memref<*xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
    %alloc = memref.alloc() : memref<8x1xf32>
    memref.copy %reinterpret_cast, %alloc : memref<8x1xf32, strided<[1, 1], offset: ?>> to memref<8x1xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<8x1xf32> to tensor<8x1xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [8, 1], strides: [1, 1] : memref<*xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<8x1xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<8x1xf32, strided<[1, 1], offset: ?>> to memref<8x1xf32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x1xf32> to tensor<8x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<8x1xf32>, tensor<8x1xf32>) outs(%2 : tensor<8x1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %5 = arith.addf %in, %in_3 : f32
      linalg.yield %5 : f32
    } -> tensor<8x1xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [8, 1], strides: [1, 1] : memref<*xf32> to memref<8x1xf32, strided<[1, 1], offset: ?>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<8x1xf32>, memref<8x1xf32, strided<[1, 1], offset: ?>>) -> ()
    return
  }
}

// -----

// Matmul op, fused with fill. No need to allocate new buffers due to having no conflicts.
// CHECK-LABEL: @func1
// CHECK:  %[[VAL_1:.*]] = bufferization.to_tensor
// CHECK:  %[[VAL_2:.*]] = bufferization.to_tensor
// CHECK-NOT:  %[[VAL_3:.*]] = bufferization.alloc_tensor

module {
  func.func @func1(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c512_i32 = arith.constant 512 : i32
    %0 = arith.muli %arg3, %c512_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %arg4, %c512_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %1, %c256 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [512, 256], strides: [256, 1] : memref<*xf32> to memref<512x256xf32, strided<[256, 1], offset: ?>>
    %alloc = memref.alloc() : memref<512x256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<512x256xf32, strided<[256, 1], offset: ?>> to memref<512x256xf32>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<512x256xf32> to tensor<512x256xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [256, 512], strides: [512, 1] : memref<*xf32> to memref<256x512xf32, strided<[512, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<256x512xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<256x512xf32, strided<[512, 1], offset: ?>> to memref<256x512xf32>
    %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<256x512xf32> to tensor<256x512xf32>
    %7 = tensor.empty() : tensor<512x512xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %9 = linalg.matmul ins(%5, %6 : tensor<512x256xf32>, tensor<256x512xf32>) outs(%8 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %10 = arith.muli %1, %c512 : index
    %11 = arith.addi %10, %3 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [512, 512], strides: [512, 1] : memref<*xf32> to memref<512x512xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<512x512xf32>, memref<512x512xf32, strided<[512, 1], offset: ?>>) -> ()
    return
  }
}
