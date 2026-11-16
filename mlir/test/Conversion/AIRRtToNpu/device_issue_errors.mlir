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

// -----

// The core's put is armed, then the host waits for an output, then a second
// put of the same core in the same run: an arm the first leaves unused could
// be taken by a start of the second before the host writes it.

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
      %q14 = arith.constant -2147483634 : i32
      memref.store %q14, %ctrl[%c2] {air.device_issue = 1 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_Out(%shim_noc_tile_1_0, S2MM, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @uncounted_two_batches(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg2 = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %0 = arith.index_cast %arg2 : index to i64
      %1 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @Out, metadata = @air_Out} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      airrt.wait_all %2
      // expected-error@+1 {{is armed after an earlier batch of its core's transfers in the same run}}
      %3 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 1 : i64, air.device_issue_bd = 14 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      affine.for %arg3 = 0 to 1 {
        %h = airrt.herd_load "router" () {segment_name = "seg"} : () -> i64
      }
      airrt.wait_all %1, %3 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// The host waits for an output before it arms the core's put: releasing the
// core after that wait could hang it, and releasing it before the arm lets it
// start on an arm its last run left.

module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %herd_lock = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %rtp = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_Out(%shim_noc_tile_1_0, S2MM, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @armed_after_wait(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %w = %c0 to %c3 step %c1 {
      %c2_i32 = arith.constant 2 : i32
      %c3_i32 = arith.constant 3 : i32
      %c0_i64 = arith.constant 0 : i64
      %layer = arith.index_cast %w : index to i32
      affine.for %arg2 = 0 to 1 {
        %p = airrt.segment_load "seg" : i64
        %0 = arith.index_cast %arg2 : index to i64
        %1 = airrt.dma_memcpy_nd(%c3_i32, %0, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @Out, metadata = @air_Out} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
        airrt.wait_all %1
        %2 = airrt.dma_memcpy_nd(%c2_i32, %0, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        affine.for %arg3 = 0 to 1 {
          // expected-error@+1 {{releases a core whose transfers are armed only after a wait in its run}}
          %h = airrt.herd_load "router" (%layer) {segment_name = "seg"} : (i32) -> i64
        }
        airrt.wait_all %2 {air.launch_end}
      } {affine_opt_label = "tiling"}
    }
    return
  }
}

// -----

// The core's put is armed inside a branch, but its release is outside it: the
// release cannot follow an arm that may not happen.

module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %herd_lock = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %rtp = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      %q14 = arith.constant -2147483634 : i32
      memref.store %q14, %ctrl[%c2] {air.device_issue = 1 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_In(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @nested(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>, %cond: i1) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    %r = arith.constant 7 : i32
    affine.for %a = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %f = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      scf.if %cond {
        %e = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        airrt.wait_all %e
      }
      affine.for %h0 = 0 to 1 {
        // expected-error@+1 {{releases a core whose transfers are armed inside a branch of its run}}
        %h = airrt.herd_load "router" (%r) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %f {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// Nothing waits inside the first wave, so both waves' puts are armed after the
// second wave's RTP write. The first wave's release can neither precede its
// arm nor follow that write.

module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %herd_lock = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %rtp = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      %q14 = arith.constant -2147483634 : i32
      memref.store %q14, %ctrl[%c2] {air.device_issue = 1 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_In(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @transition(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    %r = arith.constant 7 : i32
    affine.for %a = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %f0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      %e0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      affine.for %h0 = 0 to 1 {
        // expected-error@+1 {{releases a core whose transfers are armed only after its run}}
        %h = airrt.herd_load "router" (%r) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %f0, %e0 {air.launch_end}
    } {affine_opt_label = "tiling"}
    affine.for %a = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %f1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {chan_name = @In, metadata = @air_In} : (i32, i64, i64, memref<8x256xbf16>) : !airrt.event
      %e1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 1 : i64, air.device_issue_bd = 14 : i32, air.device_issue_count = 1 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      affine.for %h0 = 0 to 1 {
        %h = airrt.herd_load "router" (%r) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %f1, %e1 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// A second put of the core is armed in a branch after the first, so the two
// can be two batches of one run.

module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %herd_lock = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %rtp = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      %q14 = arith.constant -2147483634 : i32
      memref.store %q14, %ctrl[%c2] {air.device_issue = 1 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_In(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @cross(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>, %cond: i1) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    %r = arith.constant 7 : i32
    affine.for %a = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      // expected-note@+1 {{the earlier transfer}}
      %e = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
      scf.if %cond {
        // expected-error@+1 {{is armed in another block than an earlier transfer of its core}}
        %g = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 1 : i64, air.device_issue_bd = 14 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        airrt.wait_all %g
      }
      affine.for %h0 = 0 to 1 {
        %h = airrt.herd_load "router" (%r) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all %e {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// Both regions of a loop run on a continuing trip, so puts in its before and
// after regions are two batches of one run.

module {
  aie.device(npu2) @seg {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2)
    %herd_lock = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %rtp = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    %lock = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
    %ctrl = aie.buffer(%tile_0_2) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32>
    %core = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %q15 = arith.constant -2147483633 : i32
      memref.store %q15, %ctrl[%c2] {air.device_issue = 0 : i64} : memref<4xi32>
      %q14 = arith.constant -2147483634 : i32
      memref.store %q14, %ctrl[%c2] {air.device_issue = 1 : i64} : memref<4xi32>
      aie.end
    }
    aie.shim_dma_allocation @air_Wexp(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_In(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "router"}
    }
  }
  func.func @while_regions(%arg0: memref<32x256xbf16>, %arg1: memref<8x256xbf16>, %cond: i1) {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i64 = arith.constant 0 : i64
    %r = arith.constant 7 : i32
    affine.for %a = 0 to 1 {
      %p = airrt.segment_load "seg" : i64
      %w = scf.while (%go = %cond) : (i1) -> i1 {
        // expected-note@+1 {{the earlier transfer}}
        %e = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 0], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_count = 0 : i64, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        airrt.wait_all %e
        scf.condition(%go) %go : i1
      } do {
      ^bb0(%x: i1):
        // expected-error@+1 {{is armed in another block than an earlier transfer of its core}}
        %g = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[0, 0, 0, 256], [1, 1, 1, 256], [0, 0, 0, 1]) {air.device_issue = 1 : i64, air.device_issue_bd = 14 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2", chan_name = @Wexp, metadata = @air_Wexp} : (i32, i64, i64, memref<32x256xbf16>) : !airrt.event
        airrt.wait_all %g
        %f = arith.constant false
        scf.yield %f : i1
      }
      affine.for %h0 = 0 to 1 {
        %h = airrt.herd_load "router" (%r) {segment_name = "seg"} : (i32) -> i64
      }
      airrt.wait_all {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
