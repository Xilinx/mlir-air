//===- air_herd_to_aie_buf_names.mlir --------------------------*- MLIR -*-===//
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

// RUN: air-opt -air-to-aie %s | FileCheck %s

// CHECK-LABEL: module @aie.partition_0
// CHECK: scratch_2_2
// CHECK: buf8
// ...
// CHECK: scratch_0_0
// CHECK: buf0
func.func @launch(%arg0: i32) {
  %cst2 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
    %buf0 = memref.alloc() {sym_name = "scratch"} : memref<10xindex,2>
    %buf1 = memref.alloc() : memref<10xindex,2>
    memref.dealloc %buf0 : memref<10xindex,2>
    memref.dealloc %buf1 : memref<10xindex,2>
    air.herd_terminator
  }
  return
}
