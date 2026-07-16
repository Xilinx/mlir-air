//===- packet_l3_put_offset_dominance.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-to-aie reorders L3 (DDR) packet puts into flow order so that the runtime
// sequence emits their DMA starts in the order the receiver mem chain expects.
// The reorder relocates each put with `moveAfter(prev)`. When a put's
// wrap-and-stride offset is produced by same-block ops (here two `arith.addi`
// computing a per-iteration DDR offset) that sit between the anchor put and the
// moved put, `moveAfter` jumps the put ahead of those defining ops, leaving the
// put using a value defined after it -- an SSA dominance violation
// ("operand #N does not dominate this use") that fails the verifier.
//
// The reorder must pull the offset's same-block backward slice back before the
// put (deps-first), keeping dominance valid. This test fails to verify without
// that fix; with it, both offset `arith.addi` ops precede the put that uses them.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu1_1col' | FileCheck %s

// The offset arithmetic for @pkt_in_1 must remain before its put after reorder.
// CHECK: air.channel.put @pkt_in_0
// CHECK: arith.addi
// CHECK: arith.addi
// CHECK: air.channel.put @pkt_in_1

module {
  air.channel @pkt_in_0 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkt_in_1 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @to_core [1, 1]
  air.channel @from_core [1, 1]
  air.channel @out [1, 1]
  func.func @func_l3_put_offset(%arg0: memref<128xi32>, %arg1: memref<128xi32>, %arg2: memref<64xi32>, %iv: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    // Anchor put (flow order 0).
    air.channel.put @pkt_in_0[] (%arg0[%c0] [%c64] [%c1]) {id = 1 : i32} : (memref<128xi32>)
    // Offset arithmetic for the next put, defined BETWEEN the two puts and
    // derived from a non-constant value (%iv) so it cannot be constant-folded
    // away. After the reorder moves the put to just-after the anchor, these
    // defining ops would trail the put that uses them.
    %base = arith.addi %iv, %c1 : index
    %off = arith.addi %base, %c64 : index
    // Put (flow order 1) whose offset operand is %off.
    air.channel.put @pkt_in_1[] (%arg1[%off] [%c64] [%c1]) {id = 2 : i32} : (memref<128xi32>)
    air.segment @seg0 {
      %c1_0 = arith.constant 1 : index
      %l2_0 = memref.alloc() : memref<64xi32, 1>
      %l2_1 = memref.alloc() : memref<64xi32, 1>
      air.channel.get @pkt_in_0[] (%l2_0[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      air.channel.get @pkt_in_1[] (%l2_1[] [] []) {id = 4 : i32} : (memref<64xi32, 1>)
      air.channel.put @to_core[] (%l2_0[] [] []) {id = 5 : i32} : (memref<64xi32, 1>)
      air.herd tile(%tx, %ty) in (%sx = %c1_0, %sy = %c1_0) attributes {sym_name = "herd0"} {
        %buf = memref.alloc() : memref<64xi32, 2>
        %res = memref.alloc() : memref<64xi32, 2>
        air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 6 : i32} : (memref<64xi32, 2>)
        air.channel.put @from_core[%tx, %ty] (%res[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
        memref.dealloc %buf : memref<64xi32, 2>
        memref.dealloc %res : memref<64xi32, 2>
      }
      %l2_out = memref.alloc() : memref<64xi32, 1>
      air.channel.get @from_core[] (%l2_out[] [] []) {id = 8 : i32} : (memref<64xi32, 1>)
      air.channel.put @out[] (%l2_out[] [] []) {id = 9 : i32} : (memref<64xi32, 1>)
      memref.dealloc %l2_0 : memref<64xi32, 1>
      memref.dealloc %l2_1 : memref<64xi32, 1>
      memref.dealloc %l2_out : memref<64xi32, 1>
    }
    air.channel.get @out[] (%arg2[] [] []) {id = 10 : i32} : (memref<64xi32>)
    return
  }
}
