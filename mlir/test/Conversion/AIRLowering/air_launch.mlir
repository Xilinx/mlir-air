//===- air_launch.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-std %s | FileCheck %s

// CHECK-LABEL:   func.func @launch_0(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<16xf16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<16xf16>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           scf.parallel (%[[VAL_6:.*]], %[[VAL_7:.*]], %[[VAL_8:.*]]) = (%[[VAL_4]], %[[VAL_4]], %[[VAL_4]]) to (%[[VAL_3]], %[[VAL_2]], %[[VAL_2]]) step (%[[VAL_5]], %[[VAL_5]], %[[VAL_5]]) {
// CHECK:             air.dma_memcpy_nd (%[[VAL_1]][] [] [], %[[VAL_0]][] [] []) : (memref<16xf16>, memref<16xf16>)
func.func @launch_0(%arg0: memref<16xf16>, %arg1: memref<16xf16>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  air.launch (%arg2, %arg3, %arg4) in (%arg5=%c4, %arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg1) : memref<16xf16>, memref<16xf16> {
    air.dma_memcpy_nd (%arg9[] [] [], %arg8[] [] []) : (memref<16xf16>, memref<16xf16>)
    air.launch_terminator
  }
  return
}

// CHECK-LABEL: launch_1
// CHECK: %[[VAL_1:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[VAL_2:.*]] = airrt.wait_all %[[VAL_1]] : !airrt.event
// CHECK: %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_5:.*]] = airrt.wait_all %[[VAL_1]], %[[VAL_2]] : !airrt.event
// CHECK: scf.parallel (%[[VAL_6:.*]]) = (%[[VAL_3]]) to (%[[VAL_4]]) step (%[[VAL_4]]) {
func.func @launch_1() {
  %e0 = air.wait_all async
  %e1 = air.wait_all async [%e0]
  %t = air.launch async [%e0, %e1] () in () {
    air.launch_terminator
  }
  return
}
