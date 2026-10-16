//===- shared_device_output_elf.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-npu="output-elf=true" -verify-diagnostics

// Two funcs drive one device. output-elf names a device's sequence after the
// device, so it supports only one.

module {
  aie.device(npu2) @segment_0 {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @airMemcpyId1(%tile_0_0, MM2S, 0)
  }
  airrt.module_metadata {
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_0"} {
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 1 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd_0"}
    }
  }
  func.func @op_a(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg2 = 0 to 1 {
      %0 = airrt.segment_load "segment_0" : i64
      %1 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      %2 = airrt.wait_all %1 : !airrt.event
    } {affine_opt_label = "tiling"}
    return
  }
  // expected-error@+1 {{drives device @segment_0, as @op_a does; output-elf supports one func per device}}
  func.func @op_b(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg2 = 0 to 1 {
      %0 = airrt.segment_load "segment_0" : i64
      %1 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId1} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      %2 = airrt.wait_all %1 : !airrt.event
    } {affine_opt_label = "tiling"}
    return
  }
}
