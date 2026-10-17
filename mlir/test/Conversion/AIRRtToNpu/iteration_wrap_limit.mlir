//===- iteration_wrap_limit.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// The outermost dim of a shim BD is its iteration count. A count of 63 stays
// in one BD.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<2048x1024xbf16> offset = 0 len = 8192 sizes = [63, 8, 32, 32] strides = [32768, 32, 1024, 1])
// CHECK-NOT: aie.dma_bd

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<2048x1024xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c32_i64 = arith.constant 32 : i64
    %c63_i64 = arith.constant 63 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c63_i64, %c8_i64, %c32_i64, %c32_i64], [%c32768_i64, %c32_i64, %c1024_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<2048x1024xbf16>) : !airrt.event
    return
  }
}

// -----

// A count of 64 is split across two BDs. One BD with a count of 64 hangs on
// NPU1.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%arg0 : memref<2048x1024xbf16> offset = 0 len = 8192 sizes = [32, 8, 32, 32] strides = [32768, 32, 1024, 1])
// CHECK: aie.dma_bd(%arg0 : memref<2048x1024xbf16> offset = 1048576 len = 8192 sizes = [32, 8, 32, 32] strides = [32768, 32, 1024, 1])
// CHECK-NOT: aie.dma_bd

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<2048x1024xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c64_i64, %c8_i64, %c32_i64, %c32_i64], [%c32768_i64, %c32_i64, %c1024_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<2048x1024xbf16>) : !airrt.event
    return
  }
}
