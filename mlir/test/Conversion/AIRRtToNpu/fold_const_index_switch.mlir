//===- fold_const_index_switch.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A launch-scope scf.index_switch that selects an iteration's host feeds by a
// per-iteration mode has a CONSTANT condition once the iteration loop is
// unrolled. It must be folded away and its selected branch inlined into the
// runtime sequence -- an scf.index_switch cannot parent the emitted
// aiex.dma_configure_task_for. The switch's async drain token (threaded to the
// air.launch_end wait_all so the launch_end survives shim-dma-bds) is severed so
// the dead switch can be erased cleanly.
//
// Here the condition is constant 0, selecting the case-0 branch (@feedIn); the
// default branch (@altIn) is dead. Without the fold the scf.index_switch (and
// @altIn) survive.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// CHECK-NOT: scf.index_switch
// CHECK: aiex.dma_configure_task_for @feedIn
// CHECK-NOT: aiex.dma_configure_task_for @altIn

module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @altIn(%tile_0_0, MM2S, 1)
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
    %alt = arith.constant 5 : i32
    %sel = arith.constant 0 : index
    %ev = scf.index_switch %sel -> !airrt.event
    case 0 {
      %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      scf.yield %f : !airrt.event
    }
    default {
      %g = airrt.dma_memcpy_nd(%alt, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @altIn} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      scf.yield %g : !airrt.event
    }
    airrt.wait_all %ev {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}
