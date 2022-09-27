//===- partial_memref.mlir -------------------------------------*- MLIR -*-===//
//
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

// RUN: air-opt %s -air-dependency | FileCheck %s

// Dependency tracing capable of differentiating different pointers pointing
// to the same memref.

// CHECK-LABEL: func.func @partial_memref
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd async [
// CHECK-NOT: %[[EVENT1:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT0]]
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT0]]
// CHECK: air.herd_terminator

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
module {
  func.func @partial_memref() {
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() : memref<128x128xf32, 2>
    air.herd @herd_0  tile (%arg0, %arg1) in (%arg2=%c8, %arg3=%c2) args(%arg4=%0) : memref<128x128xf32, 2> {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c128 = arith.constant 128 : index
      %1 = affine.apply #map0()[%arg2]
      %2 = affine.apply #map1()[%arg2]
      %3 = memref.alloc() : memref<128x128xf32, 2>
      air.dma_memcpy_nd (%3[%1, %arg1] [%c16, %c32] [%c128, %c1], %arg4[%1, %arg1] [%c16, %c32] [%c128, %c1]) {id = 1 : i32} : (memref<128x128xf32, 2>, memref<128x128xf32, 2>)
      air.dma_memcpy_nd (%3[%2, %arg1] [%c16, %c32] [%c128, %c1], %arg4[%1, %arg1] [%c16, %c32] [%c128, %c1]) {id = 2 : i32} : (memref<128x128xf32, 2>, memref<128x128xf32, 2>)
      air.dma_memcpy_nd (%3[%1, %arg1] [%c16, %c32] [%c128, %c1], %arg4[%1, %arg1] [%c16, %c32] [%c128, %c1]) {id = 3 : i32} : (memref<128x128xf32, 2>, memref<128x128xf32, 2>)
      air.herd_terminator
    }
    return
  }
}