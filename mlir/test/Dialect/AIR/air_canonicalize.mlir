//===- air_herd_launch_canonicalize.mlir -----------------------*- MLIR -*-===//
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

// RUN: air-opt -canonicalize  %s | FileCheck %s

// CHECK-LABEL: func.func @herd
// CHECK: air.herd tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) {
// CHECK:   air.herd_terminator
// CHECK: }
func.func @herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}

// CHECK-LABEL: func.func @herd_async
// CHECK: air.herd async [{{.*}}] tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) {
// CHECK:   air.herd_terminator
// CHECK: }
func.func @herd_async(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  %e0 = air.wait_all async
  %e1 = air.herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: wait_all_0
// CHECK-NEXT: return
func.func @wait_all_0() {
  %0 = air.wait_all async
  %1 = air.wait_all async [%0]
  air.wait_all [%0, %1]
  air.wait_all
  return
}