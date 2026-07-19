//===- coalesce_shim_dma.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="coalesce-shim-dma=true" -split-input-file %s | FileCheck %s
// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s --check-prefix=NOCOAL

// Three consecutive contiguous MM2S transfers on the same channel, marked
// air.preserve_shim_dma_order, at contiguous source offsets 0/2048/4096 (each
// len 2048) coalesce into ONE linear BD of len 6144 -> one dma_configure_task.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<6144xbf16> offset = 0 len = 6144 sizes = [6144] strides = [1])
// CHECK-NOT: aie.dma_bd(%arg0 : memref<6144xbf16> offset = 2048
// CHECK-NOT: aie.dma_bd(%arg0 : memref<6144xbf16> offset = 4096

// With coalescing disabled the three separate 2048-length BDs remain.
// NOCOAL-LABEL: aie.device(npu2)
// NOCOAL: aie.dma_bd(%arg0 : memref<6144xbf16> offset = 0 len = 2048 sizes = [2048] strides = [1])
// NOCOAL: aie.dma_bd(%arg0 : memref<6144xbf16> offset = 2048 len = 2048 sizes = [2048] strides = [1])
// NOCOAL: aie.dma_bd(%arg0 : memref<6144xbf16> offset = 4096 len = 2048 sizes = [2048] strides = [1])

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<6144xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c2048len_i64 = arith.constant 2048 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4, air.preserve_shim_dma_order} : (i32, i64, i64, memref<6144xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c2048_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4, air.preserve_shim_dma_order} : (i32, i64, i64, memref<6144xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c4096_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4, air.preserve_shim_dma_order} : (i32, i64, i64, memref<6144xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// Transfers NOT marked air.preserve_shim_dma_order are left untouched even when
// contiguous (only lockstep-coupled paced feeds are coalesced).

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 2048 sizes = [2048] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<4096xbf16> offset = 2048 len = 2048 sizes = [2048] strides = [1])

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<4096xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<4096xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c2048_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<4096xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// A non-contiguous gap breaks the run: offsets 0 and 4096 (with a 2048 gap)
// stay as two separate BDs.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<8192xbf16> offset = 0 len = 2048 sizes = [2048] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<8192xbf16> offset = 4096 len = 2048 sizes = [2048] strides = [1])

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<8192xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4, air.preserve_shim_dma_order} : (i32, i64, i64, memref<8192xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c4096_i64], [%c1_i64, %c1_i64, %c1_i64, %c2048_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4, air.preserve_shim_dma_order} : (i32, i64, i64, memref<8192xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// Packet-switched feeds are NOT coalesced even when contiguous on the same
// channel: consecutive BDs may carry different packet ids (routing
// destinations), so merging them would misroute. The two BDs (pkt_id 1 and 2)
// stay separate; no merged len-4096 BD is produced.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 2048 sizes = [2048] strides = [1]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK: aie.dma_bd(%arg0 : memref<4096xbf16> offset = 2048 len = 2048 sizes = [2048] strides = [1]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CHECK-NOT: len = 4096

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @chA(%t, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<4096xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2048 = arith.constant 2048 : i64
    %c4 = arith.constant 4 : i32
    %0 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chA, air.preserve_shim_dma_order, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>} : (i32, i64, i64, memref<4096xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c2048], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chA, air.preserve_shim_dma_order, packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>} : (i32, i64, i64, memref<4096xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %p = airrt.segment_load "forward_0" : i64
    return
  }
}

// -----

// Feeds in different launch iterations (waves) are NOT merged across the wave
// boundary, even when their source offsets are contiguous: each wave loads its
// own slice and has its own per-wave arming that must sit between waves. The
// two iterations' feeds (offset 0 and 2048, contiguous) stay two separate BDs.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 2048 sizes = [2048] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<4096xbf16> offset = 2048 len = 2048 sizes = [2048] strides = [1])
// CHECK-NOT: len = 4096

module {
  aie.device(npu2) @seg {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %lk = aie.lock(%t02, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %rtp = aie.buffer(%t02) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
    aie.shim_dma_allocation @feedIn(%t00, MM2S, 0)
    %core = aie.core(%t02) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @ctrl(%arg0: memref<4096xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2048 = arith.constant 2048 : i64
    %c4 = arith.constant 4 : i32
    %r0 = arith.constant 7 : i32
    %r1 = arith.constant 9 : i32
    %f0 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @feedIn, air.preserve_shim_dma_order} : (i32, i64, i64, memref<4096xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %h0 = airrt.herd_load "herd_0" (%r0) {segment_name = "seg"} : (i32) -> i64
    airrt.wait_all %f0 {"air.launch_end"}
    %f1 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c2048], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @feedIn, air.preserve_shim_dma_order} : (i32, i64, i64, memref<4096xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %h1 = airrt.herd_load "herd_0" (%r1) {segment_name = "seg"} : (i32) -> i64
    airrt.wait_all %f1 {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}

// -----

// Cross-channel phase barrier: two channels (chA, chB) each contribute two
// coalesced phase runs (phase p0 offsets {0,2048}/{4096,6144}; a gap then phase
// p1 offsets {8192,10240}/{12288,14336}). Each phase per channel merges 2->1
// (len 4096). Because coalesced tasks feed distinct consumers per phase, the
// double-buffered await synthesis drains a whole phase group across channels
// before starting the next: BOTH phase-0 starts precede BOTH phase-0 awaits,
// which precede the phase-1 starts (not per-channel rolling double buffering).

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<16384xbf16> offset = 0 len = 4096
// CHECK: aiex.dma_start_task([[A0:%[0-9]+]])
// CHECK: aie.dma_bd(%arg0 : memref<16384xbf16> offset = 4096 len = 4096
// CHECK: aiex.dma_start_task([[B0:%[0-9]+]])
// CHECK: aiex.dma_await_task([[A0]])
// CHECK: aiex.dma_await_task([[B0]])
// CHECK: aie.dma_bd(%arg0 : memref<16384xbf16> offset = 8192 len = 4096
// CHECK: aiex.dma_start_task([[A1:%[0-9]+]])
// CHECK: aie.dma_bd(%arg0 : memref<16384xbf16> offset = 12288 len = 4096
// CHECK: aiex.dma_start_task([[B1:%[0-9]+]])
// CHECK: aiex.dma_await_task([[A1]])
// CHECK: aiex.dma_await_task([[B1]])

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @chA(%t, MM2S, 0)
    aie.shim_dma_allocation @chB(%t, MM2S, 1)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<16384xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2048 = arith.constant 2048 : i64
    %c4096 = arith.constant 4096 : i64
    %c6144 = arith.constant 6144 : i64
    %c8192 = arith.constant 8192 : i64
    %c10240 = arith.constant 10240 : i64
    %c12288 = arith.constant 12288 : i64
    %c14336 = arith.constant 14336 : i64
    %c4 = arith.constant 4 : i32
    %a00 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c0], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chA, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %a01 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c2048], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chA, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %b00 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c4096], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chB, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %b01 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c6144], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chB, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %a10 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c8192], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chA, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %a11 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c10240], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chA, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %b10 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c12288], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chB, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %b11 = airrt.dma_memcpy_nd(%c4, %c0, %c0, %arg0[%c0, %c0, %c0, %c14336], [%c1, %c1, %c1, %c2048], [%c0, %c0, %c0, %c1]) {metadata = @chB, air.preserve_shim_dma_order} : (i32, i64, i64, memref<16384xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %p = airrt.segment_load "forward_0" : i64
    return
  }
}
