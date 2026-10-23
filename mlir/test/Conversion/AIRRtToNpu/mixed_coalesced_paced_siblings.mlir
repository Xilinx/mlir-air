// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// RUN: air-opt -airrt-to-npu="coalesce-shim-dma=true" -split-input-file %s | FileCheck %s
//
// A repeated, strided input and a coalesced contiguous input feed the same
// consumer. Neither kind's synthesized completion drain may precede the
// sibling start. Check both emission orders and a nested scope.
// CHECK-LABEL: aie.runtime_sequence @replay_first
// CHECK: %[[FIRST:.*]] = aiex.dma_configure_task_for @replay
// CHECK: aiex.dma_start_task(%[[FIRST]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[SECOND:.*]] = aiex.dma_configure_task_for @weights
// CHECK: aiex.dma_start_task(%[[SECOND]])
// CHECK-DAG: aiex.dma_await_task(%[[FIRST]])
// CHECK-DAG: aiex.dma_await_task(%[[SECOND]])
module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @replay(%s0, MM2S, 0)
    aie.shim_dma_allocation @weights(%s1, MM2S, 0)
  } {sym_name = "projection"}
  airrt.module_metadata {}
  func.func @replay_first(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "projection" : i64
    %r = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @replay, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    airrt.wait_all %r, %w0, %w1
    return
  }
}

// -----

// CHECK-LABEL: aie.runtime_sequence @weights_first
// CHECK: %[[FIRST:.*]] = aiex.dma_configure_task_for @weights
// CHECK: aiex.dma_start_task(%[[FIRST]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[SECOND:.*]] = aiex.dma_configure_task_for @replay
// CHECK: aiex.dma_start_task(%[[SECOND]])
// CHECK-DAG: aiex.dma_await_task(%[[FIRST]])
// CHECK-DAG: aiex.dma_await_task(%[[SECOND]])
module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @replay(%s0, MM2S, 0)
    aie.shim_dma_allocation @weights(%s1, MM2S, 0)
  } {sym_name = "projection"}
  airrt.module_metadata {}
  func.func @weights_first(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "projection" : i64
    %w0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %r = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @replay, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    airrt.wait_all %r, %w0, %w1
    return
  }
}

// -----

// CHECK-LABEL: aie.runtime_sequence @loop_scope
// CHECK: %[[FIRST:.*]] = aiex.dma_configure_task_for @replay
// CHECK: aiex.dma_start_task(%[[FIRST]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[SECOND:.*]] = aiex.dma_configure_task_for @weights
// CHECK: aiex.dma_start_task(%[[SECOND]])
// CHECK-DAG: aiex.dma_await_task(%[[FIRST]])
// CHECK-DAG: aiex.dma_await_task(%[[SECOND]])
module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @replay(%s0, MM2S, 0)
    aie.shim_dma_allocation @weights(%s1, MM2S, 0)
  } {sym_name = "projection"}
  airrt.module_metadata {}
  func.func @loop_scope(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "projection" : i64
    %lo = arith.constant 0 : index
    %hi = arith.constant 3 : index
    %step = arith.constant 1 : index
    scf.for %i = %lo to %hi step %step {
      %r = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @replay, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
      %w0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
      %w1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    }
    return
  }
}

// -----

// The ordinary activation input of the second phase must be allowed to
// consume that phase's weights. The first phase must still drain first.
// CHECK-LABEL: aie.runtime_sequence @two_phases
// CHECK: %[[PHASE0:.*]] = aiex.dma_configure_task_for @weights
// CHECK: aiex.dma_start_task(%[[PHASE0]])
// CHECK: aiex.dma_await_task(%[[PHASE0]])
// CHECK: %[[ACT:.*]] = aiex.dma_configure_task_for @replay
// CHECK: aiex.dma_start_task(%[[ACT]])
// CHECK-NOT: aiex.dma_await_task(%[[ACT]])
// CHECK: %[[PHASE1:.*]] = aiex.dma_configure_task_for @weights
// CHECK: aiex.dma_start_task(%[[PHASE1]])
// CHECK-DAG: aiex.dma_await_task(%[[ACT]])
// CHECK-DAG: aiex.dma_await_task(%[[PHASE1]])
module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @replay(%s0, MM2S, 0)
    aie.shim_dma_allocation @weights(%s1, MM2S, 0)
  } {sym_name = "projection"}
  airrt.module_metadata {}
  func.func @two_phases(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "projection" : i64
    %initial0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %initial1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    airrt.wait_all %initial0, %initial1
    %r = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @replay, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @weights, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    airrt.wait_all %r, %w0, %w1
    return
  }
}
