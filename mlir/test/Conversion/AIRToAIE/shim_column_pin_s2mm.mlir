//===- shim_column_pin_s2mm.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" | FileCheck %s

// air.shim_col on the device->host (S2MM) side: two independent packet drains
// pinned to distinct columns land on separate shim tiles at those columns.
// Without the pin both drains collapse onto ONE shared column-less shim tile
// (see the pre-fix behaviour: a single `ShimNOCTile(?, ?)` carrying both
// S2MM 0 allocations), which forces two device->host streams through one shim
// column instead of the two free columns they were meant to use.

// The two pinned drains open separate shim tiles at columns 3 and 4.
// CHECK-DAG: %[[K:.*]] = aie.logical_tile<ShimNOCTile>(3, ?)
// CHECK-DAG: %[[V:.*]] = aie.logical_tile<ShimNOCTile>(4, ?)
// CHECK-DAG: aie.shim_dma_allocation @air_ka(%[[K]], S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_va(%[[V]], S2MM, 0)

air.channel @ka [1, 1] {channel_type = "npu_dma_packet", air.shim_col = 3 : i32}
air.channel @va [1, 1] {channel_type = "npu_dma_packet", air.shim_col = 4 : i32}
air.channel @tok [1, 1]
air.channel @tov [1, 1]
func.func @f(%outk: memref<64xi32>, %outv: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  air.segment @seg {
    %c1_0 = arith.constant 1 : index
    %mk = memref.alloc() : memref<64xi32, 1>
    %mv = memref.alloc() : memref<64xi32, 1>
    air.channel.get @tok[] (%mk[] [] []) {id = 1 : i32} : (memref<64xi32, 1>)
    air.channel.get @tov[] (%mv[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @ka[] (%mk[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    air.channel.put @va[] (%mv[] [] []) {id = 4 : i32} : (memref<64xi32, 1>)
    memref.dealloc %mk : memref<64xi32, 1>
    memref.dealloc %mv : memref<64xi32, 1>
    air.herd @h tile(%tx, %ty) in (%sx = %c1_0, %sy = %c1_0) {
      %b0 = memref.alloc() : memref<64xi32, 2>
      %b1 = memref.alloc() : memref<64xi32, 2>
      air.channel.put @tok[%tx, %ty] (%b0[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
      air.channel.put @tov[%tx, %ty] (%b1[] [] []) {id = 6 : i32} : (memref<64xi32, 2>)
      memref.dealloc %b0 : memref<64xi32, 2>
      memref.dealloc %b1 : memref<64xi32, 2>
    }
  }
  air.channel.get @ka[] (%outk[] [] []) {id = 7 : i32} : (memref<64xi32>)
  air.channel.get @va[] (%outv[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}
