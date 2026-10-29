//===- shim_read_back_caps.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// Six drains of one producing feed. Queue-space waits must follow that feed;
// the read-back wait follows the final drain's configure and start.
// CHECK-LABEL: aie.runtime_sequence @cap_readback
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @out
// CHECK: %[[F:.*]] = aiex.dma_configure_task_for @in
// CHECK: aiex.dma_start_task(%[[F]])
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: aiex.dma_configure_task_for @out
// CHECK: %[[D5:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_start_task(%[[D5]])
// CHECK: aiex.dma_await_task(%[[D5]])
// CHECK: aiex.dma_configure_task_for @read

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @out(%t0, S2MM, 0)
    aie.shim_dma_allocation @in(%t0, MM2S, 0)
    aie.shim_dma_allocation @read(%t1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @cap_readback(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "seg" : i64
    %d0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d5
    %r = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %f, %r
    return
  }
}

// -----

// Five chained jobs cross the four-task drain cap. Each drain must be armed
// before its producer, and each next job must wait after the preceding feed.
// CHECK-LABEL: aie.runtime_sequence @five_jobs
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @in
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: aiex.dma_configure_task_for @in
// CHECK: %[[D2:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_await_task(%[[D1]])
// CHECK: aiex.dma_configure_task_for @in
// CHECK: %[[D3:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_await_task(%[[D2]])
// CHECK: aiex.dma_configure_task_for @in
// CHECK: aiex.dma_configure_task_for @out
// CHECK: aiex.dma_await_task(%[[D3]])
// CHECK: aiex.dma_configure_task_for @in

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @out(%t0, S2MM, 0)
    aie.shim_dma_allocation @in(%t0, MM2S, 0)
    aie.shim_dma_allocation @read(%t1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @five_jobs(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "seg" : i64
    %d0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 1 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 3 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 5 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 7 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 9 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 0 : i32, metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d0
    %f1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 2 : i32, metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d1
    %f2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 4 : i32, metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d2
    %f3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 6 : i32, metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d3
    %f4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 8 : i32, metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d4, %f0, %f1, %f2, %f3, %f4
    return
  }
}

// -----

// Legacy order hints disagree with the drain channel's issue order. They must
// not swap destinations; physical S2MM channels fill buffers in issue order.
// CHECK-LABEL: aie.runtime_sequence @drain_order
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg1
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg3
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg2
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg0
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg1

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @out(%t0, S2MM, 0)
    aie.shim_dma_allocation @in(%t0, MM2S, 0)
    aie.shim_dma_allocation @read(%t1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @drain_order(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "seg" : i64
    %d0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 2 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 1 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 3 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 4 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 5 : i32, metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 0 : i32, metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d4
    %read = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 8 : i32, metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d4, %f, %read
    return
  }
}

// -----

// Conflicting and tied legacy hints must not reorder a feed channel either.
// The second A task reads the drain; all three producer B feeds precede its
// wait, and the A channel keeps its original source-buffer sequence.
// CHECK-LABEL: aie.runtime_sequence @feed_order
// CHECK: %[[FD:.*]] = aiex.dma_configure_task_for @po_out
// CHECK: aiex.dma_configure_task_for @po_a
// CHECK-NEXT: aie.dma_bd(%arg0
// CHECK: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_configure_task_for @po_b
// CHECK: aiex.dma_await_task(%[[FD]])
// CHECK: aiex.dma_configure_task_for @po_a
// CHECK-NEXT: aie.dma_bd(%arg1
// CHECK: aiex.dma_configure_task_for @po_a
// CHECK-NEXT: aie.dma_bd(%arg0

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @po_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @po_a(%t0, MM2S, 0)
    aie.shim_dma_allocation @po_b(%t1, MM2S, 0)
  } {sym_name = "po_seg"}
  airrt.module_metadata{}
  func.func @feed_order(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %d = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 0 : i32, metadata = @po_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %p = airrt.segment_load "po_seg" : i64
    %a0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 9 : i32, metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 6 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 5 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 4 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d
    %a1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 1 : i32, metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 3 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %a2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 1 : i32, metadata = @po_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %b4 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {air.program_order = 2 : i32, metadata = @po_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %a0, %a1, %a2, %b0, %b1, %b2, %b3, %b4, %d
    return
  }
}

// -----

// A read-back job's terminal output must be armed before its input feeds can
// block on output backpressure. Both drains fit in the same channel's queue,
// so the terminal drain can be armed early without a queue-cap wait.
// CHECK-LABEL: aie.runtime_sequence @terminal_output
// CHECK: %[[H:.*]] = aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg1
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg3
// CHECK: aiex.dma_configure_task_for @in
// CHECK: aiex.dma_await_task(%[[H]])
// CHECK: aiex.dma_configure_task_for @read

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @out(%t0, S2MM, 0)
    aie.shim_dma_allocation @in(%t0, MM2S, 0)
    aie.shim_dma_allocation @read(%t1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @terminal_output(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "seg" : i64
    %h = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %h
    %r = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %more0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %more1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %more2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %more3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %more4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %out = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %f, %r, %more0, %more1, %more2, %more3, %more4, %out
    return
  }
}

// -----

// A terminal drain beyond the channel cap can be armed once the earlier
// read-back retires its cap predecessor, even if the wait names a later drain
// on that channel. It must precede the next job's long input stream.
// CHECK-LABEL: aie.runtime_sequence @terminal_after_cap
// CHECK: aiex.dma_configure_task_for @in
// CHECK: aie.dma_bd(%arg3
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @read
// CHECK-COUNT-5: aiex.dma_configure_task_for @read

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @out(%t0, S2MM, 0)
    aie.shim_dma_allocation @in(%t0, MM2S, 0)
    aie.shim_dma_allocation @read(%t1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @terminal_after_cap(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "seg" : i64
    %d0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d5
    %r = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %later0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %later1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %later2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %later3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %later4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %terminal = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %f, %r, %later0, %later1, %later2, %later3, %later4, %terminal
    return
  }
}
