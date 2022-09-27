//===- air_L2cpy_to_aie.mlir -----------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=7' | FileCheck %s
// CHECK: [[T:%.*]] = AIE.tile(7, 2)
// CHECK: [[L:%.*]] = AIE.lock([[T]], {{.*}})
// CHECK: {{.*}} = AIE.mem([[T:.*]])  {
// CHECK:   AIE.useLock([[L]], Acquire, 0)
// CHECK:   AIE.dmaBd(<{{.*}} : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:   AIE.useLock([[L]], Release, 1)
// CHECK: {{.*}} = AIE.core([[T]])  {
// CHECK:   AIE.useLock([[L]], Acquire, 1)
// CHECK:   AIE.useLock([[L]], Release, 0)
// CHECK: AIE.flow({{.*}}, PLIO : 4, [[T]], DMA : 0)
module {

func.func @foo(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  %buf0 = memref.alloc() : memref<1024xi32, 1>
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %buf0, %ext1 = %arg1) : memref<1024xi32, 1>, memref<1024xi32> attributes { } {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
    %buf1 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%buf1[][][], %ext0[%c0][%c0][%c1024]) : (memref<1024xi32, 2>, memref<1024xi32, 1>)
    memref.dealloc %buf1 : memref<1024xi32, 2>
    air.herd_terminator
  }
  memref.dealloc %buf0 : memref<1024xi32, 1>
  return
}

}
