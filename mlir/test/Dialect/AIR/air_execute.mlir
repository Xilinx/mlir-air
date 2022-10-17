//===- air_execute.mlir --------------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s | FileCheck %s

func.func @air_execute_0() {

  // CHECK: air.execute -> (index)
  // CHECK: air.execute_terminator {{.*}} : index
  %1, %2 = air.execute -> (index) {
    %c1 = arith.constant 1 : index
    air.execute_terminator %c1 : index
  }

  %e = air.wait_all async
  // {{.*}} = air.execute [{{.*}}] {
  %3 = air.execute [%e] {} {foo = "bar"}

  // CHECK: air.execute -> (index)
  // CHECK: air.execute_terminator {{.*}} : index
  %4, %5 = air.execute -> (index) {
    %c1 = arith.constant 1 : index
    air.execute_terminator %c1 : index
  } {id = 1}

  // CHECK %{{.*}}, %{{.*}}:2 = air.execute [{{.*}}] -> (index, i32) {
  %6, %7, %8 = air.execute[%e, %4] -> (index, i32) {
    %c2 = arith.constant 2 : index
    %i2 = arith.constant 2 : i32
    air.execute_terminator %c2, %i2 : index, i32
  }
  return
}