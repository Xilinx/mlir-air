//===- air_multi_launch_npu.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that multiple air.launch ops are NOT merged by serializeAsyncControlFlows
// for NPU targets. The function is a VCK190-only workaround and should be skipped
// for NPU devices which use XRT.

// RUN: air-opt %s -air-to-std -canonicalize -cse --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @multi_launch_npu
// Both launches should generate their own airrt.dma_memcpy_nd operations
// with separate segment_load calls - verifying serializeAsyncControlFlows is skipped for NPU
// First launch should have its DMA ops
// CHECK: airrt.segment_load "segment_a"
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @airMemcpyId1
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @airMemcpyId2
// Second launch should have its own separate DMA ops (not merged with first launch)
// CHECK: airrt.segment_load "segment_b"
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @airMemcpyId3
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @airMemcpyId4

module {
  // NPU2 device - should skip serializeAsyncControlFlows
  aie.device(npu2) @segment_a {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId1(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, S2MM, 0)
  }
  aie.device(npu2) @segment_b {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId3(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, S2MM, 0)
  }
  airrt.module_metadata {
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_a"} {
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 1 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd_a"}
    }
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "segment_b"} {
      airrt.herd_metadata {dma_allocations = [{channel = 0 : i64, col = 0 : i64, id = 3 : i64, location = 0 : i64, row = 0 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd_b"}
    }
  }
  air.channel @channel_a_in [1, 1]
  air.channel @channel_a_out [1, 1]
  air.channel @channel_b_in [1, 1]
  air.channel @channel_b_out [1, 1]
  func.func @multi_launch_npu(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    // First launch
    %0 = air.launch async () in () args(%a0=%arg0, %a1=%arg1) : memref<64xi32>, memref<64xi32> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %2 = air.channel.put async @channel_a_in[%c0, %c0] (%a0[%c0] [%c64] [%c1]) {id = 1 : i32, metadata = @airMemcpyId1} : (memref<64xi32>)
      %3 = air.channel.get async @channel_a_out[%c0, %c0] (%a1[%c0] [%c64] [%c1]) {id = 2 : i32, metadata = @airMemcpyId2} : (memref<64xi32>)
      %4 = air.segment @segment_a async attributes {id = 2 : i32} {
        %c1_seg = arith.constant 1 : index
        %alloc_l2 = memref.alloc() : memref<64xi32, 1>
        %5 = air.channel.get async @channel_a_in[] (%alloc_l2[] [] []) {id = 11 : i32} : (memref<64xi32, 1>)
        %6 = air.channel.put async [%5] @channel_a_out[] (%alloc_l2[] [] []) {id = 12 : i32} : (memref<64xi32, 1>)
        memref.dealloc %alloc_l2 : memref<64xi32, 1>
        air.wait_all [%6] {air.segment_end}
      }
      air.wait_all [%2, %3] {air.launch_end}
    }
    // Second launch - should NOT be merged with first launch on NPU targets
    %1 = air.launch async () in () args(%a0=%arg0, %a1=%arg1) : memref<64xi32>, memref<64xi32> attributes {id = 4 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %2 = air.channel.put async @channel_b_in[%c0, %c0] (%a0[%c0] [%c64] [%c1]) {id = 3 : i32, metadata = @airMemcpyId3} : (memref<64xi32>)
      %3 = air.channel.get async @channel_b_out[%c0, %c0] (%a1[%c0] [%c64] [%c1]) {id = 4 : i32, metadata = @airMemcpyId4} : (memref<64xi32>)
      %4 = air.segment @segment_b async attributes {id = 5 : i32} {
        %c1_seg = arith.constant 1 : index
        %alloc_l2 = memref.alloc() : memref<64xi32, 1>
        %5 = air.channel.get async @channel_b_in[] (%alloc_l2[] [] []) {id = 21 : i32} : (memref<64xi32, 1>)
        %6 = air.channel.put async [%5] @channel_b_out[] (%alloc_l2[] [] []) {id = 22 : i32} : (memref<64xi32, 1>)
        memref.dealloc %alloc_l2 : memref<64xi32, 1>
        air.wait_all [%6] {air.segment_end}
      }
      air.wait_all [%2, %3] {air.launch_end}
    }
    return
  }
}
