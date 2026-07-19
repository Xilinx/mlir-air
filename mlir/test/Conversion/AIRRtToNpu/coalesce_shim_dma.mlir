//===- coalesce_shim_dma.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s
// RUN: air-opt -airrt-to-npu="coalesce-shim-dma=false" -split-input-file %s | FileCheck %s --check-prefix=NOCOAL

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
