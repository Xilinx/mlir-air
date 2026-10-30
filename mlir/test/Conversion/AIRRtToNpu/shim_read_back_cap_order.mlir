//===- shim_read_back_cap_order.mlir ----------------------------*- MLIR -*-===//
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

// Five chained jobs cross the four-task drain cap. Each drain is armed before
// its producer feed, and the wait for a drain follows that feed.
// CHECK-LABEL: aie.runtime_sequence @five_jobs
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @in
// CHECK: aiex.dma_configure_task_for @out
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK: aiex.dma_configure_task_for @out
// CHECK: aiex.dma_configure_task_for @in
// CHECK: aiex.dma_await_task
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

// A read-back job's terminal output is armed before the cap waits of the
// feeds that follow the read-back.
// CHECK-LABEL: aie.runtime_sequence @terminal_output
// CHECK: %[[H:.*]] = aiex.dma_configure_task_for @out
// CHECK: %[[F:.*]] = aiex.dma_configure_task_for @in
// CHECK: aiex.dma_await_task(%[[H]])
// CHECK: aiex.dma_configure_task_for @read
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg3
// CHECK: aiex.dma_await_task(%[[F]])
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

// A terminal drain beyond the channel cap is armed before the next job's
// input stream waits on its own cap.
// CHECK-LABEL: aie.runtime_sequence @terminal_after_cap
// CHECK: aiex.dma_configure_task_for @in
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_configure_task_for @read
// CHECK-COUNT-3: aiex.dma_configure_task_for @read
// CHECK: aiex.dma_configure_task_for @out
// CHECK-NEXT: aie.dma_bd(%arg3
// CHECK: aiex.dma_await_task
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

// -----

// Four output channels with six drains each, every channel read back by a feed
// on its own input channel. The channels share their input streams, so one
// channel's drain cannot finish while another has too few drains armed: all
// channels arm their first four drains before the first cap wait, and the cap
// waits then take turns across the channels, one drain index at a time.
// CHECK-LABEL: aie.runtime_sequence @columns
// CHECK-COUNT-4: aiex.dma_configure_task_for @out0
// CHECK-NOT: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK-NEXT: aiex.dma_configure_task_for @out0
// CHECK: aiex.dma_await_task
// CHECK-NEXT: aiex.dma_configure_task_for @out1
// CHECK: aiex.dma_await_task
// CHECK-NEXT: aiex.dma_configure_task_for @out2
// CHECK: aiex.dma_await_task
// CHECK-NEXT: aiex.dma_configure_task_for @out3
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    %t2 = aie.tile(2, 0)
    %t3 = aie.tile(3, 0)
    aie.shim_dma_allocation @out0(%t0, S2MM, 0)
    aie.shim_dma_allocation @out1(%t1, S2MM, 0)
    aie.shim_dma_allocation @out2(%t2, S2MM, 0)
    aie.shim_dma_allocation @out3(%t3, S2MM, 0)
    aie.shim_dma_allocation @in0(%t0, MM2S, 0)
    aie.shim_dma_allocation @in1(%t1, MM2S, 0)
    aie.shim_dma_allocation @in2(%t2, MM2S, 0)
    aie.shim_dma_allocation @in3(%t3, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @columns(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>, %arg3: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "seg" : i64
    %o0_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o0_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o0_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o0_3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o0_4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o0_5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1_3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1_4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o1_5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o2_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o2_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o2_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o2_3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o2_4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o2_5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o3_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o3_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o3_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o3_3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o3_4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %o3_5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @out3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f0_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f1_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f2_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f3_0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f0_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f1_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f2_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f3_1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f0_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f1_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f2_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %f3_2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o0_5
    %r0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in0} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o1_5
    %r1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o2_5
    %r2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o3_5
    %r3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @in3} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %f0_0, %f0_1, %f0_2, %f1_0, %f1_1, %f1_2, %f2_0, %f2_1, %f2_2, %f3_0, %f3_1, %f3_2, %r0, %r1, %r2, %r3
    return
  }
}
