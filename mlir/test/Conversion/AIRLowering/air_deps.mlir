//===- air_deps.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-std %s | FileCheck %s

// CHECK-LABEL: func.func @execute
// CHECK: %[[V0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
// CHECK: %[[E0:.*]] = airrt.wait_all : !airrt.event
// CHECK: airrt.wait_all %[[E0]]
// CHECK: memref.dealloc %[[V0]] : memref<64x64xi32>
// CHECK: %[[E1:.*]] = airrt.wait_all : !airrt.event
// CHECK: airrt.wait_all %[[E1]]
func.func @execute() {
  %0, %1 = air.execute -> (memref<64x64xi32>) {
    %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    air.execute_terminator %1 : memref<64x64xi32>
  }
  %2 = air.execute [%0] {
    memref.dealloc %1: memref<64x64xi32>
  }
  air.wait_all [%2]
  return
}

// CHECK-LABEL: func.func @scf_for
// CHECK: %[[V0:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[V1:.*]] = scf.for %arg0 = %c0 to %c64 step %c1 iter_args(%[[V3:.*]] = %[[V0]]) -> (!airrt.event) {
// CHECK:   %[[V2:.*]] = airrt.wait_all %[[V3]] : !airrt.event
// CHECK:   scf.yield %[[V2]] : !airrt.event
// CHECK: airrt.wait_all %[[V1]]
func.func @scf_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %0 = air.wait_all async
  %1 = scf.for %arg10 = %c0 to %c64 step %c1 iter_args(%iter_arg = %0) -> (!air.async.token) {
    %2 = air.wait_all async [%iter_arg]
    scf.yield %2 : !air.async.token
  }
  air.wait_all [%1]
  return
}

// CHECK-LABEL: func.func @scf_if
// CHECK: %[[V0:.*]] = scf.if {{.*}} -> (!airrt.event) {
// CHECK:   %[[V1:.*]] = airrt.wait_all : !airrt.event
// CHECK:   scf.yield %[[V1]] : !airrt.event
// CHECK: } else {
// CHECK:   %[[V2:.*]] = airrt.wait_all : !airrt.event
// CHECK:   scf.yield %[[V2]] : !airrt.event
// CHECK: airrt.wait_all %[[V0]]
func.func @scf_if(%0 : i1) {
  %1 = scf.if %0 -> (!air.async.token) {
    %2 = air.wait_all async
    scf.yield %2 : !air.async.token
  } else {
    %2 = air.wait_all async
    scf.yield %2 : !air.async.token
  }
  air.wait_all [%1]
  return
}
