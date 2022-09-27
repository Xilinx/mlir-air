//===- affine_opt_tile_add1.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-tile-sizes=2,2' | FileCheck %s
//
// check that it was tiled
// CHECK: affine.for %arg1 = 0 to 4 step 2 {
// CHECK:   affine.for %arg2 = 0 to 4 step 2 {
//
// check that the dma operations were generated and outlined
// CHECK: affine.dma_start %1[%arg1, %arg2], %5[%c0_4, %c0_4], %6[%c0_4], %c4_1, %c4_2, %c2_3 : memref<4x4xf32>, memref<2x2xf32, 1>, memref<1xi32>
// CHECK: affine.dma_start %7[%c0, %c0], %0[%arg1, %arg2], %8[%c0], %c4, %c4_0, %c2 : memref<2x2xf32, 1>, memref<4x4xf32>, memref<1xi32>
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