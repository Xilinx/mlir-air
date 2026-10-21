//===- shim_asymmetric_read_back.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Two herds whose channels carry very different task counts: channel A feeds
// three tasks, with a drain armed in front of the second and third; the one
// feed of channel FA reads the second drain back, behind a wait for it. The
// weave numbers a feed by its index in its channel, so the FA feed landed in
// round 0 and the second drain, held behind the first in channel order, in
// round 1: the await used a drain armed after it ("operand #0 does not
// dominate this use"). Capping the drain at its awaiter is not enough either:
// the FA feed would then wait for a drain that A's later feeds fill, before
// those feeds are issued. The FA feed goes after every feed emitted before it,
// and the drains stay where the program armed them.
// CHECK-LABEL: aie.runtime_sequence @asym_read_back
// CHECK: aiex.dma_configure_task_for @asym_a
// CHECK: aiex.dma_configure_task_for @asym_out
// CHECK: aiex.dma_configure_task_for @asym_a
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @asym_out
// CHECK: aiex.dma_configure_task_for @asym_a
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK: aiex.dma_configure_task_for @asym_fa
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t3 = aie.tile(3, 0)
    %t4 = aie.tile(4, 0)
    aie.shim_dma_allocation @asym_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @asym_a(%t3, MM2S, 0)
    aie.shim_dma_allocation @asym_fa(%t4, MM2S, 0)
  } {sym_name = "asym_seg"}
  airrt.module_metadata{}
  func.func @asym_read_back(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "asym_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %3
    %5 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @asym_fa} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %2, %4, %5
    return
  }
}
