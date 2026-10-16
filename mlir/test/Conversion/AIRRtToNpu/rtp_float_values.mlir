//===- rtp_float_values.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="output-elf=true" %s | FileCheck %s

// Float RTPs are written as their bit patterns: an f32 constant as its 32
// bits, a bf16 run-time value bitcast and zero-extended into the low half of
// its slot.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// CHECK: %[[ONE:.*]] = arith.constant 1065353216 : i32
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[ONE]]) : i32
// CHECK: %[[B:.*]] = arith.bitcast %{{.*}} : bf16 to i16
// CHECK: %[[W:.*]] = arith.extui %[[B]] : i16 to i32
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 1, %[[W]]) : i32

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<2xi32>
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

  func.func @ctrl(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %scale: bf16) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %out = arith.constant 5 : i32

    // ---- unrolled iteration 0 ----
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %one = arith.constant 1.0 : f32
    %h0 = airrt.herd_load "herd_0" (%one, %scale) {segment_name = "seg"} : (f32, bf16) -> i64
    %o0 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o0 {"air.launch_end"}

    // ---- unrolled iteration 1 ----
    %f1 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %rtp1 = arith.constant 9 : i32
    %h1 = airrt.herd_load "herd_0" (%rtp1) {segment_name = "seg"} : (i32) -> i64
    %o1 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %o1 {"air.launch_end"}

    %p = airrt.segment_load "seg" : i64
    return
  }
}
