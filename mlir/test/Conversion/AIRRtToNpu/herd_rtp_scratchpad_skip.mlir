//===- herd_rtp_scratchpad_skip.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize --split-input-file %s | FileCheck %s

// When AIRToAIE routes a herd slot through an mlir-aie scratchpad parameter,
// the core reads the parameter directly and there is nothing for this pass to
// write into the RTP buffer.
//
// The DECLARATION is the contract between the two passes, not a re-run of the
// classifier. Re-deciding "is this operand constant?" here would let the two
// disagree after any canonicalization that runs between them -- and they must
// not, because the core loads whichever source AIRToAIE picked.

// -----

// The parameter exists for slot 0, so slot 0 gets no rtp_write. The sync
// marker goes ahead of the herd's set_lock: the parameter has to be in place
// before a core is let past its herd lock, and one marker per sequence is
// enough because it syncs the whole table.

// CHECK-LABEL: aie.runtime_sequence @ctrl
// CHECK:         aiex.sync_scratchpad_parameters_from_host
// CHECK-NOT:     aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0
// CHECK:         aiex.set_lock(%{{.*}}, 1)
module {
  aiex.scratchpad_parameter @__air_param_herd_0_0 : i32
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
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

// A parameter for a DIFFERENT herd must not suppress this one's write. The
// name carries the herd symbol, so the lookup is per (herd, slot).

// CHECK-LABEL: aie.runtime_sequence @ctrl_other
// CHECK-SAME:    %{{.*}}: memref<64xi32>, %[[L:[a-zA-Z0-9_]+]]: i32
// CHECK:         aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %[[L]]) : i32
module {
  aiex.scratchpad_parameter @__air_param_someotherherd_0 : i32
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}

  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @ctrl_other(%arg0: memref<64xi32>, %L: i32) {
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

// Two slots, only the first parameterised: slot 1 still gets its write, and it
// keeps index 1. The slot counter advances for a skipped slot too, so the two
// passes' arithmetic stays aligned -- compacting here would make the core read
// the wrong word.

// CHECK-LABEL: aie.runtime_sequence @ctrl_two
// CHECK-SAME:    %{{.*}}: memref<64xi32>, %{{.*}}: i32, %[[M:[a-zA-Z0-9_]+]]: i32
// CHECK-NOT:     aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0
// CHECK:         aiex.npu.rtp_write(@__air_herd_rtp_0_2, 1, %[[M]]) : i32
module {
  aiex.scratchpad_parameter @__air_param_herd_0_0 : i32
  aie.device(npu2) @seg {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<2xi32>
    aie.shim_dma_allocation @feedIn(%tile_0_0, MM2S, 0)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {air.herd_name = "herd_0", link_with = "kernel.o"}
  } {sym_name = "seg"}

  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @ctrl_two(%arg0: memref<64xi32>, %L: i32, %M: i32) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %f0 = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %h0 = airrt.herd_load "herd_0" (%L, %M) {segment_name = "seg"} : (i32, i32) -> i64
    airrt.wait_all %f0 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}
