//===- air.mlir ------------------------------------------------*- MLIR -*-===//
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

module {

func.func @graph(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>, %arg2 : memref<4x8xi32, 1>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1, %sp0 = %arg2) : memref<16x16xi32>, memref<16x16xi32>, memref<4x8xi32, 1> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    air.dma_memcpy_nd (%sp0[%c0,%c0][%c8,%c4][%c4,%c0], %ext0[%c0,%c0][%c8,%c4][%c16,%c0]) {id = 1 : i32} : (memref<4x8xi32, 1>, memref<16x16xi32>)
    air.dma_memcpy_nd (%ext1[%c0,%c0][%c8,%c4][%c16,%c0], %sp0[%c0,%c0][%c8,%c4][%c4,%c0]) {id = 2 : i32} : (memref<16x16xi32>, memref<4x8xi32, 1>)
    air.herd_terminator
  }
  return
}

}


