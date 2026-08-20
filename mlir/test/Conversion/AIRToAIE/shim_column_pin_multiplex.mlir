//===- shim_column_pin_multiplex.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" | FileCheck %s

// Packet channels pinned to the SAME air.shim_col share one bucket keyed on
// that column, so they land on ONE shim LogicalTileOp at the pinned column
// instead of each opening its own shim there. This exercises the same-pin reuse
// path (the reuse walk restricted to LTOs on the pinned column).
//
// THREE channels, because a shim has two MM2S: multiplexing is an overflow
// mechanism, so @a and @b take the two physical channels and only @c, which has
// nowhere else to go, shares one. With two channels the pin would be satisfied
// without any multiplexing at all and the reuse path would go untested.

// Exactly one col-3 shim tile is opened; all three allocations land on it, and
// the third multiplexes onto an already-used MM2S.
// CHECK: %[[PIN:.*]] = aie.logical_tile<ShimNOCTile>(3, ?)
// CHECK-DAG: aie.shim_dma_allocation @air_a(%[[PIN]], MM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_b(%[[PIN]], MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_c(%[[PIN]], MM2S, 0)
// CHECK-NOT: aie.logical_tile<ShimNOCTile>(3,

air.channel @a [1, 1] {channel_type = "npu_dma_packet", air.shim_col = 3 : i32}
air.channel @b [1, 1] {channel_type = "npu_dma_packet", air.shim_col = 3 : i32}
air.channel @c [1, 1] {channel_type = "npu_dma_packet", air.shim_col = 3 : i32}
air.channel @ah [1, 1]
air.channel @bh [1, 1]
func.func @f(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  air.channel.put @a[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.channel.put @b[] (%arg1[] [] []) {id = 2 : i32} : (memref<64xi32>)
  air.channel.put @c[] (%arg2[] [] []) {id = 8 : i32} : (memref<64xi32>)
  air.segment @seg {
    %c1_0 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %m0 = memref.alloc() : memref<64xi32, 1>
    %m1 = memref.alloc() : memref<64xi32, 1>
    %m2 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @a[] (%m0[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    air.channel.get @b[] (%m1[] [] []) {id = 4 : i32} : (memref<64xi32, 1>)
    air.channel.get @c[] (%m2[] [] []) {id = 9 : i32} : (memref<64xi32, 1>)
    air.channel.put @ah[] (%m0[] [] []) {id = 5 : i32} : (memref<64xi32, 1>)
    air.channel.put @bh[] (%m1[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    memref.dealloc %m0 : memref<64xi32, 1>
    memref.dealloc %m1 : memref<64xi32, 1>
    memref.dealloc %m2 : memref<64xi32, 1>
    air.herd @h tile(%tx, %ty) in (%sx = %c2, %sy = %c1_0) {
      %b0 = memref.alloc() : memref<64xi32, 2>
      air.channel.get @ah[%tx, %ty] (%b0[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
      memref.dealloc %b0 : memref<64xi32, 2>
    }
  }
  return
}
