//===- air_herd_to_aie.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie | FileCheck %s
module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  // CHECK-LABEL: module @aie.segment_0
  // CHECK: %[[VAR1:.*]] = AIE.tile(1, 1)
  // CHECK: %[[BUF1:.*]] = AIE.buffer(%[[VAR1]]) {sym_name = {{.*}}} : memref<1xi32, 2>
  // CHECK: %[[BUF2:.*]] = AIE.buffer(%[[VAR1]]) {sym_name = {{.*}}} : memref<1xi32, 2>
  // CHECK: %[[BUF3:.*]] = AIE.buffer(%[[VAR1]]) {sym_name = {{.*}}} : memref<1xi32, 2>
  // CHECK: %[[VAR2:.*]] = AIE.core(%[[VAR1]])  {
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src0 = memref.alloc() : memref<1xi32, 2>
    %src1 = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    // CHECK: load %[[BUF1]]
    %0 = memref.load %src0[%zero] : memref<1xi32, 2>
    // CHECK: load %[[BUF2]]
    %1 = memref.load %src1[%zero] : memref<1xi32, 2>
    %2 = arith.addi %0, %1 :  i32
    %dst0 = memref.alloc() : memref<1xi32, 2>
    // CHECK: memref.store {{.*}}, %[[BUF3]]
    memref.store %2, %dst0[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}

}
