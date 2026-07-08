//===- rtp_setlock_region_scoping.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="output-elf=true" %s | FileCheck %s

// RTP-write / herd-release set_lock hoist is scoped to the per-load_pdi
// configuration region, NOT the global front of the runtime sequence.
//
// This is a two-iteration unrolled ELF launch: each iteration writes its own
// RTP value and re-releases the core, and the device has a repeat_count DMA so
// ELF mode inserts an aiex.npu.load_pdi reset per iteration (which re-arms the
// herd lock). A global hoist would (a) move iteration 1's RTP (9) ahead of
// iteration 0's compute, clobbering iteration 0's value (7) on the shared slot,
// and (b) move iteration 1's set_lock ahead of iteration 0's load_pdi reset,
// which clears it -> the core is never released for iteration 1 -> deadlock.
//
// Correct (region-scoped) behavior: each iteration's RTP + release sit at the
// front of THAT iteration's region -- after the preceding reset, before that
// region's data movement.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// Iteration 0: RTP (7) + release precede iteration 0's feed.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 7)
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @feedIn
// Iteration 1's RTP (9) must NOT appear before the first reset (no clobber).
// CHECK-NOT: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 9)
// CHECK: aiex.npu.load_pdi {device_ref = @seg_reset}
// Iteration 1: RTP (9) + release land AFTER the reset, before iteration 1's feed.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 9)
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @feedIn
// CHECK: aiex.npu.load_pdi {device_ref = @seg_reset}

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @outBack(%tile_0_0, S2MM, 0)
    // repeat_count DMA -> deviceNeedsLockReset -> load_pdi in ELF mode
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 3)
    ^bb1:
      aie.end
    ^bb2:
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

    // ---- unrolled iteration 0 ----
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %rtp0 = arith.constant 7 : i32
    %h0 = airrt.herd_load "herd_0" (%rtp0) {segment_name = "seg"} : (i32) -> i64
    %o0 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %o0 {"air.launch_end"}

    // ---- unrolled iteration 1 ----
    %f1 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %rtp1 = arith.constant 9 : i32
    %h1 = airrt.herd_load "herd_0" (%rtp1) {segment_name = "seg"} : (i32) -> i64
    %o1 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %o1 {"air.launch_end"}

    %p = airrt.segment_load "seg" : i64
    return
  }
}
