//===- airrt_events.mlir ------------------------------*- MLIR -*-===//
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

// RUN: air-opt -airrt-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @wait
// CHECK: %[[V0:.*]] = call @air_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK: call @air_wait_all_0_1(%[[V0]]) : (!llvm.ptr<i64>) -> ()
// CHECK: %[[V1:.*]] = call @air_wait_all_1_1(%[[V0]]) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
// CHECK: call @air_wait_all_0_2(%[[V0]], %[[V1]]) : (!llvm.ptr<i64>, !llvm.ptr<i64>) -> ()
func.func @wait() {
  %1 = airrt.wait_all : !airrt.event
  airrt.wait_all %1
  %2 = airrt.wait_all %1 : !airrt.event
  airrt.wait_all %1, %2
  return
}

// CHECK-LABEL: func.func @scf_for
// CHECK: %[[V0:.*]] = call @air_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK: %[[V1:.*]] = scf.for {{.*}} iter_args(%[[V2:.*]] = %[[V0]]) -> (!llvm.ptr<i64>) {
// CHECK:   %[[V3:.*]] = func.call @air_wait_all_1_1(%[[V2]]) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
// CHECK:   scf.yield %[[V3]] : !llvm.ptr<i64>
// CHECK: call @air_wait_all_0_1(%[[V1]]) : (!llvm.ptr<i64>) -> ()
func.func @scf_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %0 = airrt.wait_all : !airrt.event
  %1 = scf.for %arg0 = %c0 to %c64 step %c1 iter_args(%arg1 = %0) -> (!airrt.event) {
    %2 = airrt.wait_all %arg1 : !airrt.event
    scf.yield %2 : !airrt.event
  }
  airrt.wait_all %1
  return
}

// CHECK-LABEL: func.func @scf_if
// CHECK: %[[V0:.*]] = scf.if {{.*}} -> (!llvm.ptr<i64>) {
// CHECK:   %[[V1:.*]] = func.call @air_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK:   scf.yield %[[V1]] : !llvm.ptr<i64>
// CHECK: } else {
// CHECK:   %[[V1:.*]] = func.call @air_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK:   scf.yield %[[V1]] : !llvm.ptr<i64>
// CHECK: call @air_wait_all_0_1(%[[V0]]) : (!llvm.ptr<i64>) -> ()
func.func @scf_if(%arg0: i1) {
  %0 = scf.if %arg0 -> (!airrt.event) {
    %1 = airrt.wait_all : !airrt.event
    scf.yield %1 : !airrt.event
  } else {
    %1 = airrt.wait_all : !airrt.event
    scf.yield %1 : !airrt.event
  }
  airrt.wait_all %0
  return
}