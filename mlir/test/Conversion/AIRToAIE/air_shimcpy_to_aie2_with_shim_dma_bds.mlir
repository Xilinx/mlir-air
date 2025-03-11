//===- air_shimcpy_to_aie2_with_shim_dma_bds.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802 generate-shim-dma=true" -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:  %[[VAL_0:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK:  %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:  %[[VAL_2:.*]] = aie.tile(2, 0)
// CHECK:  %[[VAL_3:.*]] = aie.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_4:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {{.*}} : memref<1024xi32, 2>
// CHECK:  aie.mem(%[[VAL_1]]) {
// CHECK:    aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:    aie.dma_bd(%[[VAL_7]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:    aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:    aie.next_bd ^bb1
// CHECK:  ^bb2:
// CHECK:    aie.end
// CHECK:  }
// CHECK:  aie.core(%[[VAL_1]]) {
// CHECK:    aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:    aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:    aie.end
// CHECK:  aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:  aie.shim_dma(%[[VAL_2]]) {
// CHECK:    aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:    aie.dma_bd(%[[VAL_0]] : memref<1024xi32>, 0, 1024)
// CHECK:    aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:    aie.next_bd ^bb1
// CHECK:  ^bb2:
// CHECK:    aie.end
// CHECK:  }
func.func @func1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="herd1"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%buf0[] [] [], %ext0[%c0] [%c1024] [%c1]) : (memref<1024xi32, 2>, memref<1024xi32>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
  }
  return
}

// -----

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK: %[[VAL_0:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK: %[[VAL_1:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK: %[[VAL_2:.*]] = aie.tile(2, 3)
// CHECK: %[[VAL_3:.*]] = aie.tile(2, 0)
// CHECK: %[[VAL_4:.*]] = aie.lock(%[[VAL_3]], 3) {init = 1 : i32}
// CHECK: %[[VAL_5:.*]] = aie.lock(%[[VAL_3]], 2) {init = 0 : i32}
// CHECK: %[[VAL_6:.*]] = aie.lock(%[[VAL_3]], 1) {init = 1 : i32}
// CHECK: %[[VAL_7:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK: %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 3) {init = 1 : i32}
// CHECK: %[[VAL_9:.*]] = aie.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK: %[[VAL_10:.*]] = aie.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK: %[[VAL_11:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK: %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {{.*}} : memref<1024xi32, 2>
// CHECK: %[[VAL_13:.*]] = aie.buffer(%[[VAL_2]]) {{.*}} : memref<512xi32, 2>
// CHECK: aie.mem(%[[VAL_2]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_12]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_13]] : memref<512xi32, 2>, 0, 512)
// CHECK:   aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }
// CHECK: aie.core(%[[VAL_2]]) {
// CHECK:   aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:   aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:   aie.end
// CHECK: aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK: aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK: aie.shim_dma(%[[VAL_3]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_1]] : memref<1024xi32>, 0, 512)
// CHECK:   aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_0]] : memref<1024xi32>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }
func.func @func2(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="herd1"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    %buf1 = memref.alloc() : memref<512xi32, 2>
    air.dma_memcpy_nd (%buf0[] [] [], %ext0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)
    air.dma_memcpy_nd (%ext0[%c0] [%c512] [%c1], %buf1[] [] []) {id = 2 : i32} : (memref<1024xi32>, memref<512xi32, 2>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    memref.dealloc %buf1 : memref<512xi32, 2>
  }
  return
}

// -----

// air.channel to aie.locks.
// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:         %[[VAL_0:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_1:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 0)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_2]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_7]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_9:.*]] = aie.lock(%[[VAL_7]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_10:.*]] = aie.lock(%[[VAL_7]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_11:.*]] = aie.lock(%[[VAL_7]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_12:.*]] = aie.buffer(%[[VAL_7]]) {{{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_7]]) {{{.*}}} : memref<512xi32, 2>
// CHECK:    aie.mem(%[[VAL_7]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_12]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_13]] : memref<512xi32, 2>, 0, 512)
// CHECK:           aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:           aie.next_bd ^bb4
// CHECK:         }

// CHECK:    aie.core(%[[VAL_7]]) {
// CHECK:           aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:           aie.end
// CHECK:         }
// CHECK:         aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_7]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_7]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK: aie.shim_dma(%[[VAL_2]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_1]] : memref<1024xi32>, 0, 512)
// CHECK:   aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_0]] : memref<1024xi32>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }

air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func @func3(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.channel.put @channel_0[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="herd1"} {
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    %buf1 = memref.alloc() : memref<512xi32, 2>
    air.channel.get @channel_0[%tx, %ty] (%buf0[] [] []) {id = 2 : i32} : (memref<1024xi32, 2>)
    air.channel.put @channel_1[%tx, %ty] (%buf1[] [] []) {id = 3 : i32} : (memref<512xi32, 2>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    memref.dealloc %buf1 : memref<512xi32, 2>
  }
  air.channel.get @channel_1[] (%arg1[%c0] [%c512] [%c1]) {id = 4 : i32} : (memref<1024xi32>)
  return
}

// -----

// air.channel to aie.locks.
// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:         %[[VAL_0:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_1:.*]] = aie.external_buffer {{{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_4:.*]] = aie.tile(2, 0)
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_13:.*]] = aie.lock(%[[VAL_4]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_14:.*]] = aie.lock(%[[VAL_4]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_15:.*]] = aie.lock(%[[VAL_4]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_16:.*]] = aie.lock(%[[VAL_4]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_17:.*]] = aie.lock(%[[VAL_3]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_18:.*]] = aie.lock(%[[VAL_3]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_19:.*]] = aie.lock(%[[VAL_3]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_20:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_21:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_22:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_23:.*]] = aie.buffer(%[[VAL_3]]) {{{.*}}} : memref<1024xi32, 2>

// CHECK:    aie.mem(%[[VAL_3]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_18]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:           aie.next_bd ^bb4
// CHECK:         }

// CHECK:    aie.core(%[[VAL_3]]) {
// CHECK:           aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, 1)
// CHECK-DAG:       aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK-DAG:       aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:         aie.flow(%[[VAL_2]], DMA : 1, %[[VAL_4]], DMA : 0)
// CHECK: aie.shim_dma(%[[VAL_4]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_1]] : memref<1024xi32>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_0]] : memref<1024xi32>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }

// CHECK: aie.memtile_dma(%[[VAL_2]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb7)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(S2MM, 1, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:
// CHECK:   aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK: ^bb6:
// CHECK:   aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: ^bb7:
// CHECK:   aie.dma_start(MM2S, 1, ^bb8, ^bb5)
// CHECK: ^bb8:
// CHECK:   aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:   aie.next_bd ^bb8
// CHECK: }

air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
air.channel @channel_4 [1, 1]
air.channel @channel_5 [1, 1]
func.func @func4(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  air.channel.put @channel_2[] (%arg0[] [] []) {id = 1 : i32} : (memref<1024xi32>)
  air.segment @segment0 {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %memtile0 = memref.alloc() : memref<1024xi32, 1>
    air.channel.get @channel_2[] (%memtile0[] [] []) {id = 2 : i32} : (memref<1024xi32, 1>)
    air.channel.put @channel_3[] (%memtile0[] [] []) {id = 3 : i32} : (memref<1024xi32, 1>)
    memref.dealloc %memtile0 : memref<1024xi32, 1>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="herd4"} {
      %buf0 = memref.alloc() : memref<1024xi32, 2>
      air.channel.get @channel_3[%tx, %ty] (%buf0[] [] []) {id = 4 : i32} : (memref<1024xi32, 2>)
      air.channel.put @channel_4[%tx, %ty] (%buf0[] [] []) {id = 5 : i32} : (memref<1024xi32, 2>)
      memref.dealloc %buf0 : memref<1024xi32, 2>
    }
    %memtile1 = memref.alloc() : memref<1024xi32, 1>
    air.channel.get @channel_4[] (%memtile1[] [] []) {id = 6 : i32} : (memref<1024xi32, 1>)
    air.channel.put @channel_5[] (%memtile1[] [] []) {id = 7 : i32} : (memref<1024xi32, 1>)
    memref.dealloc %memtile1 : memref<1024xi32, 1>
  }
  air.channel.get @channel_5[] (%arg1[] [] []) {id = 8 : i32} : (memref<1024xi32>)
  return
}
