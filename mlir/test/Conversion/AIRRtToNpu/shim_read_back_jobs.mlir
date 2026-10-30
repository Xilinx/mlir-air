//===- shim_read_back_jobs.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// Chained jobs in one launch: a job writes a drain, a blocking airrt.wait_all
// waits for it, and the next job reads it back. Everything after the wait is
// issued after the drain completes, so the shim schedule may not put a blocking
// wait ahead of a transfer the awaited task depends on.

// The second job reads the drain back on channel A, then feeds five tasks on
// channel B. B's fifth task waits for B's first, which the second job only
// consumes once it has the read-back data: that pacing wait follows the
// read-back feed, not the drain's wait.
// CHECK-LABEL: aie.runtime_sequence @cross_job
// CHECK-DAG: %[[B0:.*]] = aiex.dma_configure_task_for @xj_b
// CHECK-DAG: %[[D:.*]] = aiex.dma_configure_task_for @xj_out
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK: %[[R:.*]] = aiex.dma_configure_task_for @xj_a
// CHECK-NEXT: aie.dma_bd(%arg1
// CHECK: aiex.dma_start_task(%[[R]])
// CHECK: aiex.dma_await_task(%[[B0]])
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @xj_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @xj_a(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @xj_b(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "xj_seg"}
  airrt.module_metadata{}
  func.func @cross_job(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "xj_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %2
    %5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @xj_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %3, %4, %5, %6, %7, %8, %9, %10
    return
  }
}

// -----

// Drains on one S2MM channel fill their buffers in issue order, so they keep
// their input order.
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
    %d0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg3[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d4
    %read = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @read} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %d4, %f, %read
    return
  }
}

// -----

// A feed channel keeps its input order too. The second A task reads the drain
// back; the three B feeds before its wait start ahead of that wait.
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
