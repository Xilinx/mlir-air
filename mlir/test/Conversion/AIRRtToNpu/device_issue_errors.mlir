//===- device_issue_errors.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-npu -split-input-file -verify-diagnostics

// A host feed on the shim channel of a put the core starts an uncounted
// number of times: the host waits for none of those starts, so it cannot
// order its feed after them.

module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_In(%shim_noc_tile_1_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @uncounted_shared_channel(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg2 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg2 : index to i64
      // expected-note@+1 {{the core-started transfer}}
      %1 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      // expected-error@+1 {{shares its shim channel with transfers a core starts an uncounted number of times}}
      %2 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      affine.for %arg3 = 0 to 1 {
        %h = airrt.herd_load "router" () {segment_name = "seg"} : () -> i64
      }
      airrt.wait_all %1, %2 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
