//===- rtp_setlock_region_scoping_multiiter.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// RTP-write / herd-release set_lock hoist is scoped per iteration on the fused
// multi-iteration xclbin path too -- not just the ELF (load_pdi) path.
//
// This is a two-iteration unrolled launch (two air.launch_end markers, no
// repeat_count DMA, non-ELF), so there is no aiex.npu.load_pdi to delimit the
// per-iteration configuration regions. Instead each iteration is fenced by the
// between-iteration full shim drain emitted at its launch_end, and the RTP +
// release hoist uses that drain as its region delimiter.
//
// A GLOBAL hoist would stack BOTH iterations' RTP (7 and 9) and releases at the
// front of the sequence, ahead of all data movement -- dropping the
// per-iteration re-arm, which deadlocks a finite-depth device lock after a
// couple of iterations. Correct (region-scoped) behavior: each iteration's RTP
// + release sit at the front of THAT iteration's region, after the preceding
// iteration's drain, before that region's feed.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// Iteration 0: RTP (7) + release precede iteration 0's feed.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 7)
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @feedIn
// Iteration 1's RTP (9) must NOT appear before the iteration-0 drain (no
// global hoist / no clobber).
// CHECK-NOT: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 9)
// The iteration-0 boundary drain (await on the S2MM output) fences iteration 0.
// CHECK: aiex.dma_await_task
// Iteration 1: RTP (9) + release land AFTER the drain, before iteration 1's feed.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 9)
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @feedIn

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @outBack(%tile_0_0, S2MM, 0)
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

  func.func @ctrl(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %out = arith.constant 5 : i32

    // unrolled iteration 0
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %rtp0 = arith.constant 7 : i32
    %h0 = airrt.herd_load "herd_0" (%rtp0) {segment_name = "seg"} : (i32) -> i64
    %o0 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %o0 {"air.launch_end"}

    // unrolled iteration 1
    %f1 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %rtp1 = arith.constant 9 : i32
    %h1 = airrt.herd_load "herd_0" (%rtp1) {segment_name = "seg"} : (i32) -> i64
    %o1 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %o1 {"air.launch_end"}

    %p = airrt.segment_load "seg" : i64
    return
  }
}
