//===- linalg_generic.mlir -------------------------------------*- MLIR -*-===//
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

// A single async air.execute op should be created around linalg.generic
// No air.execute op should be created inside of linalg.generic block
// CHECK: %[[EVENT0:.*]] = air.execute async [
// CHECK-NEXT: linalg.generic {
// CHECK-NEXT: ^bb0(
// CHECK-NEXT: %[[VALUE1:.*]] = linalg.index
// CHECK-NEXT: %[[VALUE2:.*]] = arith.index_cast
// CHECK-NEXT: %[[VALUE3:.*]] = arith.cmpf
// CHECK-NEXT: %[[VALUE4:.*]] = arith.select
// CHECK-NEXT: %[[VALUE5:.*]] = arith.select
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: }

#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @generic(%arg0: memref<64x64xbf16>, %arg1: memref<64x1xbf16>, %arg2: memref<64x1xi64>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2) : memref<64x64xbf16>, memref<64x1xbf16>, memref<64x1xi64> {
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map1()[%arg4]
      %2 = memref.alloc() : memref<64x64xbf16, 2>
      %3 = memref.alloc() : memref<64x1xbf16, 2>
      %4 = memref.alloc() : memref<64x1xi64, 2>
      air.dma_memcpy_nd (%2[] [] [], %arg8[%1, %c0] [%c64, %c64] [%c64, %c1_1]) {id = 1 : i32} : (memref<64x64xbf16, 2>, memref<64x64xbf16>)
      air.dma_memcpy_nd (%3[] [] [], %arg9[%1, %c0] [%c64, %c1_1] [%c1_1, %c1_1]) {id = 2 : i32} : (memref<64x1xbf16, 2>, memref<64x1xbf16>)
      air.dma_memcpy_nd (%4[] [] [], %arg10[%1, %c0] [%c64, %c1_1] [%c1_1, %c1_1]) {id = 3 : i32} : (memref<64x1xi64, 2>, memref<64x1xi64>)
      linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%2 : memref<64x64xbf16, 2>) outs(%3, %4 : memref<64x1xbf16, 2>, memref<64x1xi64, 2>) {
      ^bb0(%arg11: bf16, %arg12: bf16, %arg13: i64):
        %5 = linalg.index 1 : index
        %6 = arith.index_cast %5 : index to i64
        %7 = arith.cmpf ogt, %arg11, %arg12 : bf16
        %8 = arith.select %7, %arg11, %arg12 : bf16
        %9 = arith.select %7, %6, %arg13 : i64
        linalg.yield %8, %9 : bf16, i64
      }
      air.dma_memcpy_nd (%arg9[%1, %c0] [%c64, %c1_1] [%c1_1, %c1_1], %3[] [] []) {id = 4 : i32} : (memref<64x1xbf16>, memref<64x1xbf16, 2>)
      air.dma_memcpy_nd (%arg10[%1, %c0] [%c64, %c1_1] [%c1_1, %c1_1], %4[] [] []) {id = 5 : i32} : (memref<64x1xi64>, memref<64x1xi64, 2>)
      memref.dealloc %2 : memref<64x64xbf16, 2>
      memref.dealloc %3 : memref<64x1xbf16, 2>
      memref.dealloc %4 : memref<64x1xi64, 2>
      air.herd_terminator
    }
    return
  }
}