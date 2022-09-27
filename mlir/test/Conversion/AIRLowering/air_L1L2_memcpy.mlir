//===- air_L1L2_memcpy.mlir ------------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-to-std | FileCheck %s
// CHECK: %0 = airrt.alloc : memref<1024xi32, 1>
// CHECK: airrt.dma_memcpy_nd(%c1_i32, {{.*}}, {{.*}}, %0[{{.*}}], [{{.*}}], [{{.*}}]) : (i32, i64, i64, memref<1024xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%c2_i32, {{.*}}, {{.*}}, %0[{{.*}}], {{.*}}) : (i32, i64, i64, memref<1024xi32, 1>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dealloc %0 : memref<1024xi32, 1>
module  {
  func.func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1024xi32, 1>
    air.herd tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%0, %arg7=%arg1) : memref<1024xi32, 1>,memref<1024xi32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %1 = memref.alloc() : memref<16xi32, 2>
      air.dma_memcpy_nd (%1[][][], %arg6 [%c0] [%c16] [%c16]) {id = 1 : i32} : (memref<16xi32, 2>, memref<1024xi32, 1>)
      air.dma_memcpy_nd (%arg6[%c16] [%c0] [%c16], %1[][][]) {id = 2 : i32} : (memref<1024xi32, 1>, memref<16xi32, 2>)
      air.herd_terminator
    }
    memref.dealloc %0 : memref<1024xi32, 1>
    return
  }
}