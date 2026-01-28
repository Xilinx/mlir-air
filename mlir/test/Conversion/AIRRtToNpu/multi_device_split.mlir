//===- multi_device_split.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// Test: Multiple air.launch regions with separate aie.devices.
// Each device should get its own runtime_sequence with only the DMAs
// that belong to that device. A main orchestration device is created
// with aiex.configure and aiex.run ops.

// CHECK: aie.device(npu1_1col) @add_two
// CHECK:   aie.shim_dma_allocation @air_channel_in_add_two
// CHECK:   aie.shim_dma_allocation @air_channel_out_add_two
// CHECK:   aie.runtime_sequence @add_two_sequence(%[[ARG0:.*]]: memref<512xi32>)
// CHECK:     aiex.dma_configure_task_for @air_channel_in_add_two
// CHECK:     aiex.dma_start_task
// CHECK:     aiex.dma_configure_task_for @air_channel_out_add_two
// CHECK:     aiex.dma_start_task

// CHECK: aie.device(npu1_1col) @add_three
// CHECK:   aie.shim_dma_allocation @air_channel_in_add_three
// CHECK:   aie.shim_dma_allocation @air_channel_out_add_three
// CHECK:   aie.runtime_sequence @add_three_sequence(%{{.*}}: memref<512xi32>)
// CHECK:     aiex.dma_configure_task_for @air_channel_in_add_three
// CHECK:     aiex.dma_start_task
// CHECK:     aiex.dma_configure_task_for @air_channel_out_add_three
// CHECK:     aiex.dma_start_task

// CHECK: aie.device(npu1_1col)
// CHECK:   aie.runtime_sequence @reconfigure_example(%[[MAIN_ARG0:.*]]: memref<512xi32>)
// CHECK:     aiex.configure @add_two
// CHECK:       aiex.run @add_two_sequence(%[[MAIN_ARG0]]) : (memref<512xi32>)
// CHECK:     aiex.configure @add_three
// CHECK:       aiex.run @add_three_sequence(%[[MAIN_ARG0]]) : (memref<512xi32>)

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @air_channel_in_add_two(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_out_add_two(%tile_0_0, S2MM, 0)
  } {sym_name = "add_two"}
  
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @air_channel_in_add_three(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_out_add_three(%tile_0_0, S2MM, 0)
  } {sym_name = "add_three"}
  
  airrt.module_metadata {
  }
  
  func.func @reconfigure_example(%arg0: memref<512xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    
    // First launch region - targets add_two device
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @air_channel_in_add_two} : (i32, i64, i64, memref<512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @air_channel_out_add_two} : (i32, i64, i64, memref<512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      airrt.wait_all %0, %1
      %p = airrt.segment_load "add_two" : i64
    } {affine_opt_label = "tiling"}
    
    // Second launch region - targets add_three device
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @air_channel_in_add_three} : (i32, i64, i64, memref<512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c512_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @air_channel_out_add_three} : (i32, i64, i64, memref<512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      airrt.wait_all %0, %1
      %p = airrt.segment_load "add_three" : i64
    } {affine_opt_label = "tiling"}
    
    return
  }
}
