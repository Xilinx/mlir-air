//===- shim_column_pin_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" 2>&1 | FileCheck %s

// An air.shim_col past the device's column count is rejected rather than
// silently pinning to a nonexistent column (npu2 has 8 columns: valid [0, 8)).
// CHECK: air.shim_col column 99 is out of range [0, 8)

air.channel @a [1, 1] {channel_type = "npu_dma_packet", air.shim_col = 99 : i32}
air.channel @ah [1, 1]
func.func @f(%arg0: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  air.channel.put @a[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.segment @seg {
    %c1_0 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %m0 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @a[] (%m0[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @ah[] (%m0[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    memref.dealloc %m0 : memref<64xi32, 1>
    air.herd @h tile(%tx, %ty) in (%sx = %c2, %sy = %c1_0) {
      %b0 = memref.alloc() : memref<64xi32, 2>
      air.channel.get @ah[%tx, %ty] (%b0[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
      memref.dealloc %b0 : memref<64xi32, 2>
    }
  }
  return
}
