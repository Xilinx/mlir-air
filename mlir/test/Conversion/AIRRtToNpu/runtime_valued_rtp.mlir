//===- runtime_valued_rtp.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize --split-input-file %s | FileCheck %s

// A herd operand that is not a constant still has to reach its RTP slot. The
// core loads that slot unconditionally, so skipping the write does not leave
// the operand at some default -- it leaves the core reading stale tile memory.
// It also breaks the one invariant a runtime trip count depends on: the core's
// count and the shim's push count come from the same scalar, so if only the
// shim sees it the core hangs on a channel get.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// CHECK-SAME:    %{{.*}}: memref<64xi32>, %[[L:[a-zA-Z0-9_]+]]: i32
// CHECK:         aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[L]]) : i32
// CHECK:         aiex.set_lock(%{{.*}}, 1)
module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}

  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @ctrl(%arg0: memref<64xi32>, %L: i32) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %h0 = airrt.herd_load "herd_0" (%L) {segment_name = "seg"} : (i32) -> i64
    airrt.wait_all %f0 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// An index-typed runtime operand is narrowed to the i32 the slot holds.

// CHECK-LABEL: aie.runtime_sequence @ctrl_index
// CHECK-SAME:    %[[L:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[C:.*]] = arith.index_cast %[[L]] : index to i32
// CHECK:         aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[C]]) : i32
module {
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}

  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @ctrl_index(%arg0: memref<64xi32>, %L: index) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %h0 = airrt.herd_load "herd_0" (%L) {segment_name = "seg"} : (index) -> i64
    airrt.wait_all %f0 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}
