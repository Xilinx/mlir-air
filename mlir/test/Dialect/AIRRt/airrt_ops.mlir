//===- airrt_ops.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
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
// CHECK: %[[E0:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[E1:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[E2:.*]] = airrt.wait_all %[[E0]], %[[E1]] : !airrt.event
// CHECK: airrt.wait_all %[[E2]]
module {

func.func @f0() {
    %event1 = airrt.wait_all : !airrt.event
    %event2 = airrt.wait_all : !airrt.event
    %event3 = airrt.wait_all %event1, %event2 : !airrt.event
    airrt.wait_all %event3
    %herd_load = airrt.herd_load "herd_0" : i64
    %h, %e = airrt.herd_load "herd_0" : i64, !airrt.event
    return
}

}