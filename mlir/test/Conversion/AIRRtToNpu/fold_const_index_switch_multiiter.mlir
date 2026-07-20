//===- fold_const_index_switch_multiiter.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// Combined coverage: a fused two-iteration launch whose per-iteration feeds are
// selected by a constant-condition scf.index_switch. FoldConstIndexSwitchPattern
// inlines each switch's selected branch and rebuilds the consuming launch_end
// airrt.wait_all WITHOUT the switch token. That rebuild must preserve the
// air.launch_end marker (a discardable attr, copied via setAttrs) -- it runs
// BEFORE multi-iteration detection and wave tagging, so dropping it would
// collapse both waves' arms to the global front. Verify each wave still gets its
// OWN rtp_write (c7 then c9) armed before its own feed, and the dead branch
// (@altIn) is gone.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// CHECK-NOT: scf.index_switch
// Wave 0 arms with c7 before its feed.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %c7{{.*}}) : i32
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @outBack
// CHECK: aiex.dma_configure_task_for @feedIn
// CHECK: aiex.dma_await_task
// Wave 1 re-arms with c9 (its own arm, not collapsed to the front) before its feed.
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %c9{{.*}}) : i32
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)
// CHECK: aiex.dma_configure_task_for @outBack
// CHECK: aiex.dma_configure_task_for @feedIn
// CHECK-NOT: aiex.dma_configure_task_for @altIn

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %l = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %r = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @altIn(%tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @outBack(%tile_0_0, S2MM, 0)
    %mem = aie.mem(%tile_0_2) { aie.end }
    %core = aie.core(%tile_0_2) { aie.end } {link_with = "kernel.o"}
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
    %alt = arith.constant 5 : i32
    %out = arith.constant 6 : i32
    %sel = arith.constant 0 : index
    %rtp0 = arith.constant 7 : i32
    %rtp1 = arith.constant 9 : i32
    // iteration 0
    %ev0 = scf.index_switch %sel -> !airrt.event
    case 0 {
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      scf.yield %f : !airrt.event
    }
    default {
      %g = airrt.dma_memcpy_nd(%alt, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @altIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      scf.yield %g : !airrt.event
    }
    %h0 = airrt.herd_load "herd_0" (%rtp0) {segment_name = "seg"} : (i32) -> i64
    %o0 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %ev0, %o0 {"air.launch_end"}
    // iteration 1
    %ev1 = scf.index_switch %sel -> !airrt.event
    case 0 {
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      scf.yield %f : !airrt.event
    }
    default {
      %g = airrt.dma_memcpy_nd(%alt, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @altIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      scf.yield %g : !airrt.event
    }
    %h1 = airrt.herd_load "herd_0" (%rtp1) {segment_name = "seg"} : (i32) -> i64
    %o1 = airrt.dma_memcpy_nd(%out, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outBack} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %ev1, %o1 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}
