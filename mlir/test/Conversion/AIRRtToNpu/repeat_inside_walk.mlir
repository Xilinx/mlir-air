//===- repeat_inside_walk.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file -verify-diagnostics %s | FileCheck %s

// A stride-0 dim inside a walking dim: two row blocks, each read 4 times. A
// task repeats its whole BD, so the walking dim is split into one task per
// block, each repeating its own block.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: aie.dma_bd(%arg0 : memref<65536xbf16> offset = 0 len = 8192 sizes = [16, 512] strides = [512, 1])
// CHECK: } {repeat_count = 3 : i32}
// CHECK: aie.dma_bd(%arg0 : memref<65536xbf16> offset = 32768 len = 8192 sizes = [16, 512] strides = [512, 1])
// CHECK: } {repeat_count = 3 : i32}

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
  func.func @forward(%arg0: memref<65536xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    %c512_i64 = arith.constant 512 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c4_i64, %c16_i64, %c512_i64], [%c32768_i64, %c0_i64, %c512_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    return
  }
}

// -----

// The same split on a waited S2MM transfer: each piece is awaited once.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @airMemcpyId4
// CHECK: offset = 0 len = 8192
// CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @airMemcpyId4
// CHECK: offset = 32768 len = 8192
// CHECK: aiex.dma_await_task(%[[T0]])
// CHECK: aiex.dma_await_task(%[[T1]])

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<65536xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    %c512_i64 = arith.constant 512 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c4_i64, %c16_i64, %c512_i64], [%c32768_i64, %c0_i64, %c512_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<65536xbf16>) : !airrt.event
    airrt.wait_all %0
    return
  }
}

// -----

// A walking dim with more entries than the split allows is reported, not
// lowered with the repeat outside the walk.

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
  func.func @forward(%arg0: memref<262144xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    %c512_i64 = arith.constant 512 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c32_i64 = arith.constant 32 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    // expected-error @+1 {{a repeated dim inside a walking dim of 32 entries cannot be expressed as shim tasks}}
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c32_i64, %c4_i64, %c16_i64, %c512_i64], [%c8192_i64, %c0_i64, %c512_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<262144xbf16>) : !airrt.event
    return
  }
}
