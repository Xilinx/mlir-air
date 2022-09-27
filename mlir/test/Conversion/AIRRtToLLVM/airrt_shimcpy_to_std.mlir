//===- airrt_shimcpy_to_std.mlir -------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: call @air_dma_nd_memcpy_2d0i32(
// CHECK: call @air_dma_nd_memcpy_1d1f32(
// CHECK: call @air_dma_nd_memcpy_1d0f32(
#map0 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 128)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0 * 16)>
module  {
  func.func @ndfoo(%arg0: memref<256x256xi32>, %arg1: memref<256xf32>) {
    %L2 = airrt.alloc : memref<512xf32, 1>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        %c16 = arith.constant 16 : index
        %c0_i64 = arith.constant 0 : i64
        %c1_i64 = arith.constant 1 : i64
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c128_i64 = arith.constant 128 : i64
        %c256_i64 = arith.constant 256 : i64
        %0 = arith.index_cast %arg3 : index to i64
        %1 = arith.index_cast %arg2 : index to i64
        %2 = arith.index_cast %c16 : index to i64
        airrt.dma_memcpy_nd(%c1_i32, %0, %1, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c256_i64]) : (i32, i64, i64, memref<256x256xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
        airrt.dma_memcpy_nd(%c2_i32, %0, %1, %L2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c128_i64]) {attr = "attr"} : (i32, i64, i64, memref<512xf32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
        airrt.dma_memcpy_nd(%c2_i32, %0, %1, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %2, %2], [%c0_i64, %c0_i64, %c128_i64]) : (i32, i64, i64, memref<256xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
      } {air.herd_launch = "inner"}
    } {air.herd_launch = "outer"}
    return
  }

}

