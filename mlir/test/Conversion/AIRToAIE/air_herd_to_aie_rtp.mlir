//===- air_herd_to_aie.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie | FileCheck %s

func.func @herd1(%arg0: i32, %arg1: i32, %arg2: i32) {
  // CHECK-LABEL: aie.device
  // CHECK: %[[VAR1:.*]] = aie.tile(1, 1)
  // CHECK: %[[VAR2:.*]] = aie.tile(1, 2)

  // CHECK: %[[RTP2:.*]] = aie.buffer(%[[VAR2]]) {{{.*}}sym_name = "__air_herd_rtp_1_2"{{.*}}} : memref<3xi32>  
  // CHECK: aie.core(%[[VAR2]])
  // CHECK: load %[[RTP2]][%c0] : memref<3xi32>
  // CHECK: load %[[RTP2]][%c1] : memref<3xi32>
  // CHECK: load %[[RTP2]][%c2] : memref<3xi32>

  // CHECK: %[[RTP1:.*]] = aie.buffer(%[[VAR1]]) {{{.*}}sym_name = "__air_herd_rtp_1_1"{{.*}}} : memref<3xi32>
  // CHECK: aie.core(%[[VAR1]])
  // CHECK: load %[[RTP1]][%c0] : memref<3xi32>
  // CHECK: load %[[RTP1]][%c1] : memref<3xi32>
  // CHECK: load %[[RTP1]][%c2] : memref<3xi32>
  %cst1 = arith.constant 1 : index
  %cst2 = arith.constant 2 : index
  %cst12 = arith.constant 12 : i32
  %cst23 = arith.constant 23 : i32
  %cst34 = arith.constant 34 : i32
  air.herd @herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst2) args(%a = %cst12, %b = %cst23, %c = %cst34) : i32, i32, i32 {
    %src0 = memref.alloc() : memref<1xi32, 2>
    %src1 = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    %0 = memref.load %src0[%zero] : memref<1xi32, 2>
    %1 = memref.load %src1[%zero] : memref<1xi32, 2>
    %2 = arith.addi %0, %a : i32
    %3 = arith.addi %1, %b : i32
    %4 = arith.addi %2, %3 : i32
    %5 = arith.addi %4, %c : i32
    %dst0 = memref.alloc() : memref<1xi32, 2>
    memref.store %5, %dst0[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}

