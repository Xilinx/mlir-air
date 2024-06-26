//===- air_herd_to_aie_sizes.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-aie %s | FileCheck %s

func.func @launch(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  // CHECK: %[[TILE01:.*]] = aie.tile(1, 2)
  // CHECK: {{.*}} = aie.core(%[[TILE01]])  {
  // CHECK: memref.store {{.*}}, {{.*}}[{{.*}}] : memref<1024xindex, 2>
  // CHECK: aie.end
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
    %buf = memref.alloc() : memref<1024xindex,2>
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    memref.store %0, %buf[%1] : memref<1024xindex,2>
  }
  return
}
