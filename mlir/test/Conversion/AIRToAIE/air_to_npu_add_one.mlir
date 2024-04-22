//===- air_to_npu_spatial_add_one.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(func.func(convert-linalg-to-affine-loops), air-to-aie{row-offset=2 col-offset=0 device=npu})' --split-input-file | FileCheck %s

// CHECK: %[[VAL0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL2:.*]] = aie.tile(0, 0)
// CHECK: %[[VAL3:.*]] = aie.lock(%[[VAL0]], 3) {init = 1 : i32}
// CHECK: %[[VAL4:.*]] = aie.lock(%[[VAL0]], 2) {init = 0 : i32}
// CHECK: %[[VAL5:.*]] = aie.lock(%[[VAL0]], 1) {init = 1 : i32}
// CHECK: %[[VAL6:.*]] = aie.lock(%[[VAL0]], 0) {init = 0 : i32}
// CHECK: %[[VAL7:.*]] = aie.lock(%[[VAL1]], 3) {init = 1 : i32}
// CHECK: %[[VAL8:.*]] = aie.lock(%[[VAL1]], 2) {init = 0 : i32}
// CHECK: %[[VAL9:.*]] = aie.lock(%[[VAL1]], 1) {init = 1 : i32}
// CHECK: %[[VAL10:.*]] = aie.lock(%[[VAL1]], 0) {init = 0 : i32}
// CHECK: %[[VAL11:.*]] = aie.buffer(%[[VAL0]]) {sym_name = {{.*}}} : memref<64xi32, 1>
// CHECK: %[[VAL12:.*]] = aie.buffer(%[[VAL0]]) {sym_name = {{.*}}} : memref<64xi32, 1>
// CHECK: %[[VAL13:.*]] = aie.buffer(%[[VAL1]]) {sym_name = {{.*}}} : memref<64xi32, 2>
// CHECK: %[[VAL14:.*]] = aie.buffer(%[[VAL1]]) {sym_name = {{.*}}} : memref<64xi32, 2>
// CHECK: aie.mem(%[[VAL1]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL7]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL13]] : memref<64xi32, 2>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL8]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:  // pred: ^bb0
// CHECK:   aie.dma_start(MM2S, 0, ^bb4,
// CHECK: ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:   aie.use_lock(%[[VAL10]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL14]] : memref<64xi32, 2>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL9]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }
// CHECK: aie.core(%[[VAL1]]) {
// CHECK:   %[[VAL15:.*]] = arith.constant 1 : i32
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   aie.use_lock(%[[VAL9]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL8]], AcquireGreaterEqual, 1)
// CHECK:   affine.for %[[VAL16:.*]] = 0 to 64 {
// CHECK:     %[[VAL17:.*]] = affine.load %[[VAL13]][%[[VAL16]]] : memref<64xi32, 2>
// CHECK:     %[[VAL18:.*]] = arith.addi %[[VAL17]], %[[VAL15]] : i32
// CHECK:     affine.store %[[VAL18]], %[[VAL14]][%[[VAL16]]] : memref<64xi32, 2>
// CHECK:   }
// CHECK:   aie.use_lock(%[[VAL7]], Release, 1)
// CHECK:   aie.use_lock(%[[VAL10]], Release, 1)
// CHECK:   aie.end
// CHECK: }
// CHECK: aie.flow(%[[VAL2]], DMA : 0, %[[VAL0]], DMA : 0)
// CHECK: aie.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: aie.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: aie.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: aie.memtile_dma(%[[VAL0]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL11]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL6]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:
// CHECK:   aie.dma_start(S2MM, 1, ^bb4
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL3]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL12]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL4]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:
// CHECK:   aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
// CHECK: ^bb6:
// CHECK:   aie.use_lock(%[[VAL6]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL11]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL5]], Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: ^bb7:
// CHECK:   aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
// CHECK: ^bb8:
// CHECK:   aie.use_lock(%[[VAL4]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL12]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL3]], Release, 1)
// CHECK:   aie.next_bd ^bb8
// CHECK: }
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: @func0
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
      air.herd_terminator
    }
    %memtile1 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @channel_2[] (%memtile1[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    air.channel.put @channel_3[] (%memtile1[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
    memref.dealloc %memtile1 : memref<64xi32, 1>
    air.segment_terminator
  }
  air.channel.get @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}

// -----

// Asynchronous version

// CHECK: %[[VAL0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL2:.*]] = aie.tile(0, 0)
// CHECK: %[[VAL3:.*]] = aie.lock(%[[VAL0]], 3) {init = 1 : i32}
// CHECK: %[[VAL4:.*]] = aie.lock(%[[VAL0]], 2) {init = 0 : i32}
// CHECK: %[[VAL5:.*]] = aie.lock(%[[VAL0]], 1) {init = 1 : i32}
// CHECK: %[[VAL6:.*]] = aie.lock(%[[VAL0]], 0) {init = 0 : i32}
// CHECK: %[[VAL7:.*]] = aie.lock(%[[VAL1]], 3) {init = 1 : i32}
// CHECK: %[[VAL8:.*]] = aie.lock(%[[VAL1]], 2) {init = 0 : i32}
// CHECK: %[[VAL9:.*]] = aie.lock(%[[VAL1]], 1) {init = 1 : i32}
// CHECK: %[[VAL10:.*]] = aie.lock(%[[VAL1]], 0) {init = 0 : i32}
// CHECK: %[[VAL11:.*]] = aie.buffer(%[[VAL0]]) {sym_name = {{.*}}} : memref<64xi32, 1>
// CHECK: %[[VAL12:.*]] = aie.buffer(%[[VAL0]]) {sym_name = {{.*}}} : memref<64xi32, 1>
// CHECK: %[[VAL13:.*]] = aie.buffer(%[[VAL1]]) {sym_name = {{.*}}} : memref<64xi32, 2>
// CHECK: %[[VAL14:.*]] = aie.buffer(%[[VAL1]]) {sym_name = {{.*}}} : memref<64xi32, 2>
// CHECK: aie.mem(%[[VAL1]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL7]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL13]] : memref<64xi32, 2>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL8]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:  // pred: ^bb0
// CHECK:   aie.dma_start(MM2S, 0, ^bb4,
// CHECK: ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:   aie.use_lock(%[[VAL10]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL14]] : memref<64xi32, 2>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL9]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }
// CHECK: aie.core(%[[VAL1]]) {
// CHECK:   %[[VAL15:.*]] = arith.constant 1 : i32
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   aie.use_lock(%[[VAL9]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL8]], AcquireGreaterEqual, 1)
// CHECK:   affine.for %[[VAL16:.*]] = 0 to 64 {
// CHECK:     %[[VAL17:.*]] = affine.load %[[VAL13]][%[[VAL16]]] : memref<64xi32, 2>
// CHECK:     %[[VAL18:.*]] = arith.addi %[[VAL17]], %[[VAL15]] : i32
// CHECK:     affine.store %[[VAL18]], %[[VAL14]][%[[VAL16]]] : memref<64xi32, 2>
// CHECK:   }
// CHECK:   aie.use_lock(%[[VAL7]], Release, 1)
// CHECK:   aie.use_lock(%[[VAL10]], Release, 1)
// CHECK:   aie.end
// CHECK: }
// CHECK: aie.flow(%[[VAL2]], DMA : 0, %[[VAL0]], DMA : 0)
// CHECK: aie.flow(%[[VAL0]], DMA : 0, %[[VAL1]], DMA : 0)
// CHECK: aie.flow(%[[VAL1]], DMA : 0, %[[VAL0]], DMA : 1)
// CHECK: aie.flow(%[[VAL0]], DMA : 1, %[[VAL2]], DMA : 0)
// CHECK: aie.memtile_dma(%[[VAL0]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL11]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL6]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:
// CHECK:   aie.dma_start(S2MM, 1, ^bb4
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL3]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL12]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL4]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:
// CHECK:   aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
// CHECK: ^bb6:
// CHECK:   aie.use_lock(%[[VAL6]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL11]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL5]], Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: ^bb7:
// CHECK:   aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
// CHECK: ^bb8:
// CHECK:   aie.use_lock(%[[VAL4]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL12]] : memref<64xi32, 1>, 0, 64)
// CHECK:   aie.use_lock(%[[VAL3]], Release, 1)
// CHECK:   aie.next_bd ^bb8
// CHECK: }
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: @func1
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
        air.herd_terminator
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
      air.segment_terminator
    }
    %2 = air.channel.get async  @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
    return
  }
}
