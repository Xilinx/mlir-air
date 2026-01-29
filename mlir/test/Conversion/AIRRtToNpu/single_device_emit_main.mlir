//===- single_device_emit_main.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that a single aie.device design generates a "main" aie.device wrapper
// when --emit-main-device=true is specified. This enables reconfiguration mode
// even for single-device designs.

// RUN: air-opt %s -airrt-to-npu="emit-main-device=true" --split-input-file | FileCheck %s

// CHECK-LABEL: module
// CHECK: aie.device(npu2) @segment_0
// CHECK:   aie.runtime_sequence @segment_0_sequence
// The main device should be created with configure/run ops
// CHECK: aie.device(npu2)
// CHECK:   aie.runtime_sequence @single_launch
// CHECK:     aiex.configure @segment_0
// CHECK:       aiex.run @segment_0_sequence

module {
  aie.device(npu2) @segment_0 {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @airMemcpyId1(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, S2MM, 0)
  }
  airrt.module_metadata {
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_0"} {
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 1 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd_0"}
    }
  }
  air.channel @channel_in [1, 1]
  air.channel @channel_out [1, 1]
  func.func @single_launch(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    // Single launch with segment_load
    affine.for %arg2 = 0 to 1 {
      %0 = airrt.segment_load "segment_0" : i64
      %1 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId1} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      %3 = airrt.wait_all %1, %2 : !airrt.event
    } {affine_opt_label = "tiling"}
    return
  }
}

// -----

// Test that single device WITHOUT emit-main-device does NOT generate main device wrapper
// The function should be moved directly into the device as aie.runtime_sequence

// RUN: air-opt %s -airrt-to-npu --split-input-file | FileCheck %s --check-prefix=NO-MAIN

// NO-MAIN-LABEL: module
// NO-MAIN: aie.device(npu2) @segment_0
// NO-MAIN:   aie.runtime_sequence @single_launch_no_main
// Verify there is only one aie.device (no main device)
// NO-MAIN-NOT: aiex.configure
// NO-MAIN-NOT: aiex.run

module {
  aie.device(npu2) @segment_0 {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @airMemcpyId1(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, S2MM, 0)
  }
  airrt.module_metadata {
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_0"} {
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 1 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd_0"}
    }
  }
  air.channel @channel_in [1, 1]
  air.channel @channel_out [1, 1]
  func.func @single_launch_no_main(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    // Single launch with segment_load
    affine.for %arg2 = 0 to 1 {
      %0 = airrt.segment_load "segment_0" : i64
      %1 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId1} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      %3 = airrt.wait_all %1, %2 : !airrt.event
    } {affine_opt_label = "tiling"}
    return
  }
}
