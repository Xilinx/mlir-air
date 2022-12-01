//===- affine_opt_tile_add1.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-tile-sizes=2,2' | FileCheck %s
//
// check that it was tiled
// CHECK: affine.for %arg1 = 0 to 4 step 2 {
// CHECK:   affine.for %arg2 = 0 to 4 step 2 {
//
// check that the dma operations were generated and outlined
// CHECK: affine.dma_start %{{.*}}[%arg1, %arg2], %{{.*}}[%c0_4, %c0_4], %{{.*}}[%c0_4], %c4_1, %c4_2, %c2_3 : memref<4x4xf32>, memref<2x2xf32, 1>, memref<1xi32>
// CHECK: affine.dma_start %{{.*}}[%c0, %c0], %{{.*}}[%arg1, %arg2], %{{.*}}[%c0], %c4, %c4_0, %c2 : memref<2x2xf32, 1>, memref<4x4xf32>, memref<1xi32>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (4)>
module {
  func.func @graph(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = arith.constant 1.0 : f32
    %1 = memref.alloc() : memref<4x4xf32>
    %2 = bufferization.to_memref %arg0 : memref<4x4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c4_0 = arith.constant 4 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c4_3 = arith.constant 4 : index
    %c4_4 = arith.constant 4 : index
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    affine.for %arg1 = 0 to 4 {
      affine.for %arg2 = 0 to 4 {
        %4 = affine.load %2[%arg1, %arg2] : memref<4x4xf32>
        %cst = arith.constant 1.000000e+00 : f32
        %5 = arith.addf %4, %cst : f32
        affine.store %4, %1[%arg1, %arg2] : memref<4x4xf32>
      }
    } {affine_opt_label = "affine_opt"}
    %3 = bufferization.to_tensor %1 : memref<4x4xf32>
    return %3 : tensor<4x4xf32>
  }
}