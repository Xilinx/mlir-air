//===- air_herd_to_aie_buf_names.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-aie %s | FileCheck %s

// CHECK: aie.device
// CHECK: scratch_2_2
// CHECK: buf8
// ...
// CHECK: scratch_0_0
// CHECK: buf0
func.func @launch(%arg0: i32) {
  %cst2 = arith.constant 3 : index
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
    %buf0 = memref.alloc() {sym_name = "scratch"} : memref<10xindex,2>
    %buf1 = memref.alloc() : memref<10xindex,2>
    memref.dealloc %buf0 : memref<10xindex,2>
    memref.dealloc %buf1 : memref<10xindex,2>
  }
  return
}
