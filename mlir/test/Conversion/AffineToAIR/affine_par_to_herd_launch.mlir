//===- affine_par_to_herd_launch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-par-to-herd %s | FileCheck %s
// CHECK-LABEL: func.func @foo
// CHECK: %[[C0:.*]] = arith.constant 1 : index
// CHECK air.herd tile ({{.*}}, {{.*}}) in ({{.*}}=[[C0]], {{.*}}=[[C0]])
module  {
  func.func @foo()  {
    affine.parallel (%x,%y) = (0,0) to (1,1) {
      %2 = arith.addi %x, %y : index
      affine.yield
    }
    return
  }
}