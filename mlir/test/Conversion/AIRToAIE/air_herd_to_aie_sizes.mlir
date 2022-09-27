//===- air_herd_to_aie_sizes.mlir ------------------------------*- MLIR -*-===//
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

func.func @launch(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  // CHECK: %[[TILE01:.*]] = AIE.tile(0, 1)
  // CHECK: {{.*}} = AIE.core(%[[TILE01]])  {
  // CHECK: memref.store {{.*}}, {{.*}}[{{.*}}] : memref<1024xindex, 2>
  // CHECK: AIE.end
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
    %buf = memref.alloc() : memref<1024xindex,2>
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    memref.store %0, %buf[%1] : memref<1024xindex,2>
    air.herd_terminator
  }
  return
}
