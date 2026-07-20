//===- fused_multiiter_output_s2mm_first.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A fused multi-iteration launch (two air.launch_end markers, non-ELF) must arm
// each iteration's output S2MM BEFORE issuing that iteration's input MM2S feed.
// The producing core blocks releasing its output buffer lock until the output
// S2MM BD is running to drain it; if the control program is still issuing input
// feeds at that point the core stalls -> deadlock. So each iteration's output
// S2MM configure+start is hoisted ahead of its first input feed.
//
// The two iterations are HETEROGENEOUS: iteration 0 also drains an extra output
// (@appendOut) that iteration 1 does not, exercising a per-iteration output set
// that is not uniform across waves.
//
// Without the hoist the input feed (@feedIn) is configured first, and the extra
// per-wave output has no output-first ordering.

// CHECK-LABEL: aie.runtime_sequence @ctrl

// Iteration 0: arm, then BOTH outputs (@appendOut, @outBack) configure+start
// BEFORE the input feed @feedIn.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %{{.*}}) : i32
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @appendOut
// CHECK: aiex.dma_configure_task_for @outBack
// CHECK: aiex.dma_configure_task_for @feedIn
// CHECK: aiex.dma_await_task

// Iteration 1: re-arm lands AFTER iteration 0's drain (no global hoist), and its
// single output @outBack is configured before @feedIn. @appendOut is absent.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %{{.*}}) : i32
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @outBack
// CHECK-NOT: aiex.dma_configure_task_for @appendOut
// CHECK: aiex.dma_configure_task_for @feedIn

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @outBack(%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @appendOut(%tile_0_0, S2MM, 1)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}

  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @ctrl(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %out = arith.constant 5 : i32
    %app = arith.constant 6 : i32

    // unrolled iteration 0 (heterogeneous: also drains @appendOut)
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %rtp0 = arith.constant 7 : i32
    %h0 = airrt.herd_load "herd_0" (%rtp0) {segment_name = "seg"} : (i32) -> i64
    %a0 = airrt.dma_memcpy_nd(%app, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendOut} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %o0 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %o0 {"air.launch_end"}

    // unrolled iteration 1 (no @appendOut)
    %f1 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %rtp1 = arith.constant 9 : i32
    %h1 = airrt.herd_load "herd_0" (%rtp1) {segment_name = "seg"} : (i32) -> i64
    %o1 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %o1 {"air.launch_end"}

    %p = airrt.segment_load "seg" : i64
    return
  }
}
