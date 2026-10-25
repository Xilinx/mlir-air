//===- paced_coalesced_sibling_drains.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="coalesce-shim-dma=true" -split-input-file %s | FileCheck %s

// A paced feed (@act: strided, so never coalesced) and a coalesced phase
// (@wts: two contiguous halves merged into one task) on sibling channels. A
// paced feed belongs to the first phase that ends after it, or to the last
// phase when issued after every phase; a feed and the phase it belongs to both
// start before either is awaited. A phase still drains before the next phase
// starts. A paced fragment on a coalesced channel is not a sibling.

// CHECK-LABEL: aie.runtime_sequence @paced_then_phase
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[W]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @paced_then_phase(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    return
  }
}

// -----

// CHECK-LABEL: aie.runtime_sequence @phase_then_paced
// CHECK: %[[W:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[W]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @phase_then_paced(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    return
  }
}

// -----

// A loop body is its own segment.
// CHECK-LABEL: aie.runtime_sequence @paced_in_loop
// CHECK: scf.for
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[W]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @paced_in_loop(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %lo = arith.constant 0 : index
    %hi = arith.constant 3 : index
    %st = arith.constant 1 : index
    scf.for %i = %lo to %hi step %st {
      %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
      %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
      %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    }
    return
  }
}

// -----

// The feed belongs to the later phase; the earlier one drains before it.
// CHECK-LABEL: aie.runtime_sequence @paced_between_phases
// CHECK: %[[W0:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W0]])
// CHECK: aiex.dma_await_task(%[[W0]])
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W1:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W1]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[W1]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @paced_between_phases(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w1a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    return
  }
}

// -----

// Issued after every phase, the feed belongs to the last one.
// CHECK-LABEL: aie.runtime_sequence @paced_after_last_phase
// CHECK: %[[W0:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W0]])
// CHECK: aiex.dma_await_task(%[[W0]])
// CHECK: %[[W1:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W1]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[W1]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @paced_after_last_phase(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    return
  }
}

// -----

// The feed drains with the phase it belongs to, before the next phase starts.
// CHECK-LABEL: aie.runtime_sequence @paced_before_middle_phase
// CHECK: %[[W0:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W0]])
// CHECK: aiex.dma_await_task(%[[W0]])
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W1:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W1]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])
// CHECK-DAG: aiex.dma_await_task(%[[W1]])
// CHECK: %[[W2:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W2]])
// CHECK: aiex.dma_await_task(%[[W2]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @paced_before_middle_phase(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w1a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w2a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w2b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    return
  }
}

// -----

// The run drains before its own channel's trailing fragment starts; @act is
// still drained after the segment's last paced start.
// CHECK-LABEL: aie.runtime_sequence @fragment_after_sibling
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W]])
// CHECK: aiex.dma_await_task(%[[W]])
// CHECK-NOT: aiex.dma_await_task(%[[A]])
// CHECK: %[[F:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[F]])
// CHECK-DAG: aiex.dma_await_task(%[[F]])
// CHECK-DAG: aiex.dma_await_task(%[[A]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @fragment_after_sibling(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %a = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %f = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c16, %c8, %c512], [%c0, %c4096, %c512, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    return
  }
}

// -----

// Per-channel pacing still bounds @act to two tasks in flight; only its tail
// waits for the phase to start.
// CHECK-LABEL: aie.runtime_sequence @paced_burst_then_phase
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_await_task(%[[A0]])
// CHECK: aiex.dma_start_task(%[[A2]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W]])
// CHECK-DAG: aiex.dma_await_task(%[[A1]])
// CHECK-DAG: aiex.dma_await_task(%[[A2]])
// CHECK-DAG: aiex.dma_await_task(%[[W]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @paced_burst_then_phase(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %a0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %a1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %a2 = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    return
  }
}

// -----

// Each launch wave of a fused launch is its own segment: wave 0 drains before
// wave 1 starts.
// CHECK-LABEL: aie.runtime_sequence @two_waves
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W0:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W0]])
// CHECK-DAG: aiex.dma_await_task(%[[A0]])
// CHECK-DAG: aiex.dma_await_task(%[[W0]])
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @act
// CHECK: aiex.dma_start_task(%[[A1]])
// CHECK-NOT: aiex.dma_await_task
// CHECK: %[[W1:.*]] = aiex.dma_configure_task_for @wts
// CHECK: aiex.dma_start_task(%[[W1]])
// CHECK-DAG: aiex.dma_await_task(%[[A1]])
// CHECK-DAG: aiex.dma_await_task(%[[W1]])

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    aie.shim_dma_allocation @act(%s0, MM2S, 0)
    aie.shim_dma_allocation @wts(%s1, MM2S, 0)
  } {sym_name = "seg"}
  airrt.module_metadata {}
  func.func @two_waves(%x: memref<65536xbf16>, %w: memref<131072xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c16 = arith.constant 16 : i64
    %c512 = arith.constant 512 : i64
    %c4096 = arith.constant 4096 : i64
    %c65536 = arith.constant 65536 : i64
    %id = arith.constant 0 : i32
    %p = airrt.segment_load "seg" : i64
    %a0 = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w0a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w0b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    airrt.wait_all %a0, %w0a, %w0b {"air.launch_end"}
    %a1 = airrt.dma_memcpy_nd(%id, %c0, %c0, %x[%c0, %c0, %c0, %c0], [%c16, %c8, %c16, %c512], [%c0, %c512, %c4096, %c1]) {metadata = @act, air.preserve_shim_dma_order} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    %w1a = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    %w1b = airrt.dma_memcpy_nd(%id, %c0, %c0, %w[%c0, %c0, %c0, %c65536], [%c1, %c1, %c1, %c65536], [%c0, %c0, %c0, %c1]) {metadata = @wts, air.preserve_shim_dma_order} : (i32, i64, i64, memref<131072xbf16>) : !airrt.event
    airrt.wait_all %a1, %w1a, %w1b {"air.launch_end"}
    return
  }
}
