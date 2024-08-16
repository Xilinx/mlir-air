//===- air_to_npu_spatial_add_one.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(air-to-aie{row-offset=2 col-offset=0 device=npu1_1col insert-control-packet-flow=true})' --split-input-file | FileCheck %s

// CHECK: %[[VAL0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK:   aie.packet_source<%[[VAL2]], DMA : 0>
// CHECK:   aie.packet_dest<%[[VAL0]], DMA : 0>
// CHECK: }
// CHECK: aie.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: aie.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: aie.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: @func0
// CHECK: air.channel.put  @channel_0[] {{.*}} metadata = @airMemcpyId2, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
#map2 = affine_map<(d0) -> (d0)>
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
func.func @func0(%arg0 : memref<64xi32>, %arg1 : memref<64xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  air.channel.put @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.segment @segment0 {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %memtile0 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_0[] (%memtile0[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_1[] (%memtile0[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile0 : memref<64xi32, 1>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="func4"} {
      %buf0 = memref.alloc() : memref<64xi32, 2>
      %buf1 = memref.alloc() : memref<64xi32, 2>
      air.channel.get @channel_1[%tx, %ty] (%buf0[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
      linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%buf0 : memref<64xi32, 2>) outs(%buf1 : memref<64xi32, 2>) {
      ^bb0(%arg11: i32, %arg12: i32):
        %c1_32 = arith.constant 1 : i32
        %12 = arith.addi %arg11, %c1_32 : i32
        linalg.yield %12 : i32
      }
      air.channel.put @channel_2[%tx, %ty] (%buf1[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
      memref.dealloc %buf0 : memref<64xi32, 2>
      memref.dealloc %buf1 : memref<64xi32, 2>
    }
    %memtile1 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_2[] (%memtile1[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_3[] (%memtile1[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile1 : memref<64xi32, 1>
  }
  air.channel.get @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}

// -----

// Asynchronous version

// CHECK: %[[VAL0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK:   aie.packet_source<%[[VAL2]], DMA : 0>
// CHECK:   aie.packet_dest<%[[VAL0]], DMA : 0>
// CHECK: }
// CHECK: aie.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: aie.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: aie.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: @func1
// CHECK: air.channel.put async @channel_0[] {{.*}} metadata = @airMemcpyId2, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>
#map = affine_map<(d0) -> (d0)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func1(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %0 = air.channel.put async  @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
    %1 = air.segment @segment0 async  attributes {id = 2 : i32} {
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 1 : i32}
      %3 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
      %4 = air.channel.put async [%3]  @channel_1[] (%results[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      %async_token_2 = air.execute [%4] {
        memref.dealloc %results : memref<64xi32, 1>
      } {id = 2 : i32}
      %5 = air.herd @func4 async  tile (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_1) attributes {id = 1 : i32} {
        %async_token_6, %results_7 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 3 : i32}
        %async_token_8, %results_9 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 4 : i32}
        %8 = air.channel.get async [%async_token_6]  @channel_1[%arg2, %arg3] (%results_7[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        %async_token_10 = air.execute [%async_token_8, %8] {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%results_7 : memref<64xi32, 2>) outs(%results_9 : memref<64xi32, 2>) {
          ^bb0(%in: i32, %out: i32):
            %c1_i32 = arith.constant 1 : i32
            %10 = arith.addi %in, %c1_i32 : i32
            linalg.yield %10 : i32
          }
        } {id = 5 : i32}
        %9 = air.channel.put async [%async_token_10]  @channel_2[%arg2, %arg3] (%results_9[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
        %async_token_11 = air.execute [%async_token_10] {
          memref.dealloc %results_7 : memref<64xi32, 2>
        } {id = 6 : i32}
        %async_token_12 = air.execute [%9] {
          memref.dealloc %results_9 : memref<64xi32, 2>
        } {id = 7 : i32}
      }
      %async_token_3, %results_4 = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 8 : i32}
      %6 = air.channel.get async [%async_token_3]  @channel_2[] (%results_4[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
      %7 = air.channel.put async [%6]  @channel_3[] (%results_4[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
      %async_token_5 = air.execute [%7] {
        memref.dealloc %results_4 : memref<64xi32, 1>
      } {id = 9 : i32}
    }
    %2 = air.channel.get async  @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
    return
  }
}
