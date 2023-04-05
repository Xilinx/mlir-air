//===- scf_parallel_to_launch_and_segment.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-par-to-launch='has-air-segment=true' -cse -canonicalize %s | FileCheck %s
// CHECK-LABEL: func.func @f0
// CHECK: %[[C0:.*]] = arith.constant 2 : index
// CHECK air.launch (%[[V0:.*]], %[[V1:.*]]) in (%[[V2:.*]]=[[C0]], %[[V3:.*]]=[[C0]])
// CHECK air.segment args({{.*}}=[[V0]], {{.*}}=[[V1]], {{.*}}=[[V2]], {{.*}}=[[V3]])
func.func @f0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0,%c0) to (%c2, %c2) step (%c1,%c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @f1
// CHECK: %[[C1:.*]] = arith.constant 4 : index
// CHECK air.launch (%[[V0:.*]]) in (%[[V1:.*]]=[[C1]])
// CHECK air.segment args({{.*}}=[[V0]], {{.*}}=[[V1]])
func.func @f1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %2 = arith.muli %x, %x : index
    scf.yield
  }
  return
}
