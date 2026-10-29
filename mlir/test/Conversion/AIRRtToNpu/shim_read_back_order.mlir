//===- shim_weave_program_order.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// Two jobs share feed channels A and B. The first job's three B feeds must
// all start before the drain wait guarding the second job's first A feed.
// Task order is the input's issue order; static operation IDs are not used.
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
    %d = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "po_seg" : i64
    %a0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d
    %a1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b4 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %a0, %a1, %a2, %b0, %b1, %b2, %b3, %b4, %d
    return
  }
}

// -----

// A channel used only by the next job must not overtake that job's read-back
// barrier. Otherwise its fifth task's queue-cap wait can block on a producer
// that still needs the reader behind the barrier.
// CHECK-LABEL: aie.runtime_sequence @later_feeds
// CHECK: %[[LD:.*]] = aiex.dma_configure_task_for @po_out
// CHECK: aiex.dma_configure_task_for @po_a
// CHECK-COUNT-6: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_await_task(%[[LD]])
// CHECK: %[[READER:.*]] = aiex.dma_configure_task_for @po_a
// CHECK: aiex.dma_start_task(%[[READER]])
// CHECK-COUNT-5: aiex.dma_configure_task_for @po_c

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t2 = aie.tile(2, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @po_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @po_a(%t0, MM2S, 0)
    aie.shim_dma_allocation @po_c(%t2, MM2S, 0)
    aie.shim_dma_allocation @po_b(%t1, MM2S, 0)
  } {sym_name = "po_seg"}
  airrt.module_metadata{}
  func.func @later_feeds(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %d = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "po_seg" : i64
    %a0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b4 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b5 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d
    %a1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %cc0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_c} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %cc1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_c} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %cc2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_c} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %cc3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_c} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %cc4 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @po_c} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %a0, %a1, %b0, %b1, %b2, %b3, %b4, %b5, %cc0, %cc1, %cc2, %cc3, %cc4
    return
  }
}
