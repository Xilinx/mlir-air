//===- shim_weave_program_order.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// The conversion hands the weave its transfers regrouped by channel: every task
// for A, then every task for B. Two jobs share A and B here. The first has one
// A task and three B tasks and drains into %arg1; the second reads %arg1 back
// through its first A task, behind a wait for the drain. The weave numbers a feed
// by its index in its channel, so that A task (index 1) used to go out in round 1,
// ahead of the first job's last B task (round 2) -- but the drain only completes
// once that B task has been fed. With air.program_order on the tasks the jobs
// are told apart again: the wait starts a new segment, and everything the drain
// needs goes out before it.
// CHECK-LABEL: aie.runtime_sequence @jobs
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @po_out
// CHECK: aiex.dma_configure_task_for @po_a
// CHECK: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK: aiex.dma_configure_task_for @po_a
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @po_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @po_a(%t0, MM2S, 0)
    aie.shim_dma_allocation @po_b(%t1, MM2S, 0)
  } {sym_name = "po_seg"}
  airrt.module_metadata{}
  func.func @jobs(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %d = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 5 : i32, metadata = @po_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "po_seg" : i64
    %a0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 1 : i32, metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d
    %a1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 6 : i32, metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 8 : i32, metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 2 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 3 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 4 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 7 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b4 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 9 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %a0, %a1, %a2, %b0, %b1, %b2, %b3, %b4, %d
    return
  }
}
