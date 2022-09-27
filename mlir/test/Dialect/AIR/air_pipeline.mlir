//===- air_pipeline.mlir ---------------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s | FileCheck %s
module  {
  func.func @launch(%arg0: i32) {
    %c4 = arith.constant 4 : index
    air.herd tile (%arg1, %arg2) in (%arg3=%c4, %arg4=%c4) args(%arg5=%arg0, %arg6=%arg0) : i32,i32 {
      %c1_i32 = arith.constant 1 : i32
      // CHECK: air.pipeline attributes {direction = "horiz"} {
      air.pipeline attributes {direction = "horiz"} {
        // CHECK: %{{.*}} = air.pipeline.stage args(%{{.*}}=%{{.*}}) : i32 {
        // CHECK: air.pipeline.yield %{{.*}} : i32
        // CHECK: } : i32
        %output1 = air.pipeline.stage args(%in = %c1_i32) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        } : i32
        %output2 = air.pipeline.stage args(%in = %output1) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        } : i32
        %output3 = air.pipeline.stage args(%in = %output2) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        } : i32
        air.pipeline.stage args(%in = %output3) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        }
        air.pipeline.terminator
      }
      air.herd_terminator
    }
    return
  }
}
