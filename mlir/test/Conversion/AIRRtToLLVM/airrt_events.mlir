//===- airrt_events.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-llvm %s | FileCheck %s

// CHECK-LABEL: func.func @wait
// CHECK: %[[V0:.*]] = call @__airrt_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK: call @__airrt_wait_all_0_1(%[[V0]]) : (!llvm.ptr<i64>) -> ()
// CHECK: %[[V1:.*]] = call @__airrt_wait_all_1_1(%[[V0]]) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
// CHECK: call @__airrt_wait_all_0_2(%[[V0]], %[[V1]]) : (!llvm.ptr<i64>, !llvm.ptr<i64>) -> ()
func.func @wait() {
  %1 = airrt.wait_all : !airrt.event
  airrt.wait_all %1
  %2 = airrt.wait_all %1 : !airrt.event
  airrt.wait_all %1, %2
  return
}

// CHECK-LABEL: func.func @scf_for
// CHECK: %[[V0:.*]] = call @__airrt_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK: %[[V1:.*]] = scf.for {{.*}} iter_args(%[[V2:.*]] = %[[V0]]) -> (!llvm.ptr<i64>) {
// CHECK:   %[[V3:.*]] = func.call @__airrt_wait_all_1_1(%[[V2]]) : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
// CHECK:   scf.yield %[[V3]] : !llvm.ptr<i64>
// CHECK: call @__airrt_wait_all_0_1(%[[V1]]) : (!llvm.ptr<i64>) -> ()
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
// CHECK:   %[[V1:.*]] = func.call @__airrt_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK:   scf.yield %[[V1]] : !llvm.ptr<i64>
// CHECK: } else {
// CHECK:   %[[V1:.*]] = func.call @__airrt_wait_all_1_0() : () -> !llvm.ptr<i64>
// CHECK:   scf.yield %[[V1]] : !llvm.ptr<i64>
// CHECK: call @__airrt_wait_all_0_1(%[[V0]]) : (!llvm.ptr<i64>) -> ()
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