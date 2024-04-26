//===- air_shimcpy_to_aie2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(2, 3)
// CHECK:  %[[VAL_1:.*]] = aie.tile(2, 0)
// CHECK:  %[[VAL_2:.*]] = aie.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_3:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK:  %[[VAL_5:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:    %[[VAL_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:    aie.dma_bd(%[[VAL_4]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:    aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:    aie.next_bd ^bb1
// CHECK:  ^bb2:
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %[[VAL_7:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:    aie.use_lock(%[[VAL_2]], Release, 1)
// CHECK:    aie.end
// CHECK:  aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:  aie.shim_dma_allocation @airMemcpyId0(MM2S, 0, 2)
// CHECK: @func1
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
    air.herd_terminator
  }
  return
}

// -----

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK: %[[VAL_0:.*]] = aie.tile(2, 3)
// CHECK: %[[VAL_1:.*]] = aie.tile(2, 0)
// CHECK: %[[VAL_2:.*]] = aie.lock(%[[VAL_0]], 3) {init = 1 : i32}
// CHECK: %[[VAL_3:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32}
// CHECK: %[[VAL_4:.*]] = aie.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK: %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK: %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK: %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<512xi32, 2>
// CHECK: %[[VAL_8:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:   %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_7]] : memref<512xi32, 2>, 0, 512)
// CHECK:   aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: }
// CHECK: %[[VAL_11:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL_2]], Release, 1)
// CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:   aie.end
// CHECK: aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK: aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId2(S2MM, 0, 2)
// CHECK: memref.global "public" @airMemcpyId2 : memref<512xi32, 2>
// CHECK: aie.shim_dma_allocation @airMemcpyId1(MM2S, 0, 2)
// CHECK: memref.global "public" @airMemcpyId1 : memref<1024xi32, 2>
// CHECK: @func2
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
    air.herd_terminator
  }
  return
}

// -----

// air.channel to aie.locks.
// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.lock(%[[VAL_1]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_6:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<512xi32, 2>

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
// CHECK:         ^bb4:
// CHECK:           aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_7]] : memref<512xi32, 2>, 0, 512)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.next_bd ^bb4
// CHECK:         }

// CHECK:    aie.core(%[[VAL_1]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_2]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         aie.shim_dma_allocation @airMemcpyId3(S2MM, 0, 2)
// CHECK:         memref.global "public" @airMemcpyId3 : memref<512xi32, 2>
// CHECK:         aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 2)
// CHECK:         memref.global "public" @airMemcpyId2 : memref<1024xi32, 2>
// CHECK: @func3
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
    air.herd_terminator
  }
  air.channel.get @channel_1[] (%arg1[%c0] [%c512] [%c1]) {id = 4 : i32} : (memref<1024xi32>)
  return
}

// -----

// air.channel to aie.locks.
// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_4:.*]] = aie.tile(2, 0)
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_17:.*]] = aie.lock(%[[VAL_3]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_18:.*]] = aie.lock(%[[VAL_3]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_19:.*]] = aie.lock(%[[VAL_3]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_20:.*]] = aie.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_21:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_22:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_23:.*]] = aie.buffer(%[[VAL_3]]) {{{.*}}} : memref<1024xi32, 2>

// CHECK:    aie.mem(%[[VAL_3]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_18]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
// CHECK:         ^bb4:
// CHECK:           aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:           aie.next_bd ^bb4
// CHECK:         }

// CHECK:    aie.core(%[[VAL_3]]) {
// CHECK:           aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:         aie.flow(%[[VAL_2]], DMA : 1, %[[VAL_4]], DMA : 0)

// CHECK: aie.memtile_dma(%[[VAL_2]]) {
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:
// CHECK:   aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
// CHECK: ^bb6:
// CHECK:   aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: ^bb7:
// CHECK:   aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
// CHECK: ^bb8:
// CHECK:   aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:   aie.next_bd ^bb8
// CHECK: }
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 2)
// CHECK: memref.global "public" @airMemcpyId7 : memref<1024xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 2)
// CHECK: memref.global "public" @airMemcpyId2 : memref<1024xi32, 1>
// CHECK: @func4
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
      air.herd_terminator
    }
    %memtile1 = memref.alloc() : memref<1024xi32, 1>
    air.channel.get @channel_4[] (%memtile1[] [] []) {id = 6 : i32} : (memref<1024xi32, 1>)
    air.channel.put @channel_5[] (%memtile1[] [] []) {id = 7 : i32} : (memref<1024xi32, 1>)
    memref.dealloc %memtile1 : memref<1024xi32, 1>
    air.segment_terminator
  }
  air.channel.get @channel_5[] (%arg1[] [] []) {id = 8 : i32} : (memref<1024xi32>)
  return
}

// -----

// L2 to L1 broadcast
// CHECK: aie.device
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK:         %[[VAL_4:.*]] = aie.tile(4, 3)
// CHECK:         %[[VAL_5:.*]] = aie.tile(5, 3)
// CHECK:         %[[VAL_6:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_7:.*]] = aie.tile(3, 4)
// CHECK:         %[[VAL_8:.*]] = aie.tile(4, 4)
// CHECK:         %[[VAL_9:.*]] = aie.tile(5, 4)
// CHECK:         %[[VAL_10:.*]] = aie.tile(2, 5)
// CHECK:         %[[VAL_11:.*]] = aie.tile(3, 5)
// CHECK:         %[[VAL_12:.*]] = aie.tile(4, 5)
// CHECK:         %[[VAL_13:.*]] = aie.tile(5, 5)
// CHECK:         %[[VAL_14:.*]] = aie.tile(2, 6)
// CHECK:         %[[VAL_15:.*]] = aie.tile(3, 6)
// CHECK:         %[[VAL_16:.*]] = aie.tile(4, 6)
// CHECK:         %[[VAL_17:.*]] = aie.tile(5, 6)

// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_10]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_14]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_2]], DMA : 1)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_4]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_5]], DMA : 0)

// CHECK:         aie.shim_dma_allocation @airMemcpyId6(MM2S, 0, 2)
// CHECK:         memref.global "public" @airMemcpyId6 : memref<1024xi32, 1>
// CHECK: @func5
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
air.channel @channel_6 [1, 1] {broadcast_shape = [1, 4]}
air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
air.channel @channel_8 [1, 1]
func.func @func5(%arg0 : memref<1024xi32>) -> () {
  %token_0 = air.channel.put async @channel_8[] (%arg0[] [] []) {id = 3 : i32} : (memref<1024xi32>)
  %token_10 = air.segment @segment0 async {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %herd_cols = arith.constant 4 : index
    %herd_rows = arith.constant 4 : index
    %token_1, %memtile0 = air.execute -> (memref<1024xi32, 1>) {
      %alloc = memref.alloc() : memref<1024xi32, 1>
      air.execute_terminator %alloc : memref<1024xi32, 1>
    }
    %token_2 = air.channel.get async [%token_1] @channel_8[] (%memtile0[] [] []) {id = 6 : i32} : (memref<1024xi32, 1>)
    %token_3 = air.channel.put async [%token_2] @channel_6[] (%memtile0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32, 1>)
    %token_4 = air.channel.put async [%token_3] @channel_7[] (%memtile0[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32, 1>)
    %token_5 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="herd5"} {
      %token_6, %buf0 = air.execute -> (memref<1024xi32, 2>) {
        %alloc = memref.alloc() : memref<1024xi32, 2>
        air.execute_terminator %alloc : memref<1024xi32, 2>
      }
      %token_7, %buf1 = air.execute -> (memref<512xi32, 2>) {
        %alloc = memref.alloc() : memref<512xi32, 2>
        air.execute_terminator %alloc : memref<512xi32, 2>
      }
      %aif0 = affine.if #set()[%tx, %ty] -> !air.async.token {
        %17 = air.channel.get async [%token_6, %token_7]  @channel_6[%tx, %ty] (%buf0[] [] []) {id = 3 : i32} : (memref<1024xi32, 2>)
        affine.yield %17 : !air.async.token
      } else {
        %17 = air.wait_all async [%token_6, %token_7]
        affine.yield %17 : !air.async.token
      }
      %aif1 = affine.if #set1()[%tx, %ty] -> !air.async.token {
        %17 = air.channel.get async [%aif0]  @channel_7[%tx, %ty] (%buf1[] [] []) {id = 4 : i32} : (memref<512xi32, 2>)
        affine.yield %17 : !air.async.token
      } else {
        %17 = air.wait_all async [%aif0]
        affine.yield %17 : !air.async.token
      }
      %token_8 = air.execute [%aif1] {
        memref.dealloc %buf0 : memref<1024xi32, 2>
      }
      %token_9 = air.execute [%aif1] {
        memref.dealloc %buf1 : memref<512xi32, 2>
      }
      air.herd_terminator
    }
    air.segment_terminator
  }
  return
}

// -----

// L3 to L1 parallel shim dmas
// CHECK: aie.device(xcve2802)
// CHECK: %[[tile_2_0:.*]] = aie.tile(2, 0)
// CHECK: %[[tile_3_0:.*]] = aie.tile(3, 0)
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK: %[[tile_1_3:.*]] = aie.tile(1, 3)
// CHECK: %[[tile_0_4:.*]] = aie.tile(0, 4)
// CHECK: %[[tile_1_4:.*]] = aie.tile(1, 4)

// CHECK:  aie.flow(%[[tile_0_3]], DMA : 0, %[[tile_2_0]], DMA : 0)
// CHECK:  aie.flow(%[[tile_1_3]], DMA : 0, %[[tile_2_0]], DMA : 1)
// CHECK:  aie.flow(%[[tile_0_4]], DMA : 0, %[[tile_3_0]], DMA : 0)
// CHECK:  aie.flow(%[[tile_1_4]], DMA : 0, %[[tile_3_0]], DMA : 1)
// CHECK:  aie.shim_dma_allocation @airMemcpyId14(S2MM, 0, 2)
// CHECK:  memref.global "public" @airMemcpyId14 : memref<4x4xi32, 2>
// CHECK:  aie.shim_dma_allocation @airMemcpyId14_1(S2MM, 1, 2)
// CHECK:  memref.global "public" @airMemcpyId14_1 : memref<4x4xi32, 2>
// CHECK:  aie.shim_dma_allocation @airMemcpyId14_2(S2MM, 0, 3)
// CHECK:  memref.global "public" @airMemcpyId14_2 : memref<4x4xi32, 2>
// CHECK:  aie.shim_dma_allocation @airMemcpyId14_3(S2MM, 1, 3)
// CHECK:  memref.global "public" @airMemcpyId14_3 : memref<4x4xi32, 2>

// CHECK: @func6
// CHECK: air.channel.get{{.*}}metadata = @airMemcpyId14} : (memref<8x8xi32>)
#map1 = affine_map<()[s0] -> (s0 * 4)>
air.channel @channel_0 [2, 2]
func.func @func6(%arg5 : memref<8x8xi32>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) args(%arg4=%arg5) : memref<8x8xi32> attributes {id = 1 : i32} {
    %c0_8 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_7 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %3 = air.wait_all async 
    %4 = scf.parallel (%arg6, %arg7) = (%c0_8, %c0_8) to (%c2, %c2) step (%c1_7, %c1_7) init (%3) -> !air.async.token {
      %async_token_17, %results_18 = air.execute -> (index) {
        %7 = affine.apply #map1()[%arg6]
        air.execute_terminator %7 : index
      }
      %async_token_19, %results_20 = air.execute -> (index) {
        %7 = affine.apply #map1()[%arg7]
        air.execute_terminator %7 : index
      }
      %6 = air.channel.get async [%async_token_19, %async_token_17]  @channel_0[%arg6, %arg7] (%arg4[%results_18, %results_20] [%c4, %c4] [%c8, %c1_7]) {id = 3 : i32} : (memref<8x8xi32>)
      scf.reduce(%6 : !air.async.token) {
      ^bb0(%arg8: !air.async.token, %arg9: !air.async.token):
        %7 = air.wait_all async [%arg8, %arg9] 
        scf.reduce.return %7 : !air.async.token
      }
    }
    %5 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 2 : i64, y_loc = 3 : i64, y_size = 2 : i64} {
      %c2_22 = arith.constant 2 : index
      %25 = air.herd @herd_0 async tile (%arg6, %arg7) in (%arg8=%c2_22, %arg9=%c2_22) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
        %async_token_34, %results_35 = air.execute -> (memref<4x4xi32, 2>) {
          %alloc = memref.alloc() : memref<4x4xi32, 2>
          air.execute_terminator %alloc : memref<4x4xi32, 2>
        }
        %27 = air.channel.put async [%async_token_34]  @channel_0[%arg6, %arg7] (%results_35[] [] []) {id = 14 : i32} : (memref<4x4xi32, 2>)
        %async_token_45 = air.execute [%27] {
          memref.dealloc %results_35 : memref<4x4xi32, 2>
        }
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// Multi-dimensional memref copy to wraps and strides
// CHECK: aie.device(xcve2802)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
// CHECK: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<4x4xi32, 1>, 0, 16)
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:  // pred: ^bb5
// CHECK:   aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
// CHECK: ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<8x16xi32, 1>, 0, 16, [<size = 4, stride = 16>, <size = 4, stride = 1>])
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:  // pred: ^bb0
// CHECK:   aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
// CHECK: ^bb6:  // 2 preds: ^bb5, ^bb6
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<4x4xi32, 1>, 0, 16)
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: @func7
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
air.channel @channel_2 [1, 1]
func.func @func7(%arg0 : memref<8x16xi32>, %arg1 : memref<16x8xi32>){
  air.channel.put @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<8x16xi32>)
  air.segment args(%ext0 = %arg0, %ext1 = %arg1) : memref<8x16xi32>, memref<16x8xi32> attributes {sym_name="segment", id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 3 : i64, y_size = 1 : i64} {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    air.herd @herd_0 tile (%arg6, %arg7) in (%arg8=%herd_cols, %arg9=%herd_rows) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
      %results_35 = memref.alloc() : memref<4x4xi32, 2>
      air.channel.put @channel_2[%arg6, %arg7] (%results_35[] [] []) {id = 14 : i32} : (memref<4x4xi32, 2>)
      memref.dealloc %results_35 : memref<4x4xi32, 2>
      air.herd_terminator
    }
    %buf0 = memref.alloc() : memref<4x4xi32, 1>
    %buf1 = memref.alloc() : memref<8x16xi32, 1>
    air.channel.get @channel_0[] (%buf0[] [] []) {id = 2 : i32} : (memref<4x4xi32, 1>)
    air.channel.put @channel_1[] (%buf0[] [] []) {id = 3 : i32} : (memref<4x4xi32, 1>)
    air.channel.get @channel_2[] (%buf1[%c0, %c0] [%c4, %c4] [%c16, %c1]) {id = 4 : i32} : (memref<8x16xi32, 1>)
    memref.dealloc %buf0 : memref<4x4xi32, 1>
    memref.dealloc %buf1 : memref<8x16xi32, 1>
    air.segment_terminator
  }
  air.channel.get @channel_1[] (%arg1[] [] []) {id = 4 : i32} : (memref<16x8xi32>)
  return
}

// -----

// Multi-dimensional memref copy to wraps and strides, with offsets having more dims than memref type.
// CHECK: aie.device(xcve2802)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<64x256xi32, 1>, 8192, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:  // pred: ^bb0
// CHECK:   aie.end
// CHECK: @func8
air.channel @channel_0 [1, 1]
func.func @func8(%arg0 : memref<8x16xi32>, %arg1 : memref<16x8xi32>){
  air.segment args(%ext0 = %arg0, %ext1 = %arg1) : memref<8x16xi32>, memref<16x8xi32> attributes {sym_name="segment", id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 3 : i64, y_size = 1 : i64} {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    air.herd @herd_0 tile (%arg6, %arg7) in (%arg8=%herd_cols, %arg9=%herd_rows) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
      %results_35 = memref.alloc() : memref<32x32xi32, 2>
      air.channel.get @channel_0[%arg6, %arg7] (%results_35[] [] []) {id = 14 : i32} : (memref<32x32xi32, 2>)
      memref.dealloc %results_35 : memref<32x32xi32, 2>
      air.herd_terminator
    }
    %buf0 = memref.alloc() : memref<64x256xi32, 1>
    air.channel.put @channel_0[] (%buf0[%c0, %c32, %c0] [%c8, %c32, %c32] [%c32, %c256, %c1]) : (memref<64x256xi32, 1>)
    memref.dealloc %buf0 : memref<64x256xi32, 1>
    air.segment_terminator
  }
  return
}
// -----

// 1D scf.parallel iteration space support.
// CHECK: aie.device(xcve2802)
// CHECK: %[[tileDMA_0_4:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:   aie.dma_bd({{.*}} : memref<32xf32, 2>, 0, 32)
// CHECK: %[[tileDMA_0_3:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:   aie.dma_bd({{.*}} : memref<32xf32, 2>, 0, 32)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK:   aie.dma_bd({{.*}} : memref<64xf32, 1>, 0, 32)
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2, repeat_count = 1)
// CHECK:   aie.dma_bd({{.*}} : memref<64xf32, 1>, 32, 32)
// CHECK: @func9
#map = affine_map<()[s0] -> (s0 * 32)>
air.channel @channel_1 [2, 1]
func.func @func9(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg2) in (%arg3=%c2) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 3 : i64, y_size = 2 : i64} {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<64xf32, 1>) {
        %alloc = memref.alloc() : memref<64xf32, 1>
        air.execute_terminator %alloc : memref<64xf32, 1>
      }
      %2 = scf.parallel (%arg4) = (%c0) to (%c2_0) step (%c1) init (%async_token) -> !air.async.token {
        %async_token_2, %results_3 = air.execute -> (index) {
          %5 = affine.apply #map()[%arg4]
          air.execute_terminator %5 : index
        }
        %4 = air.channel.put async [%async_token]  @channel_1[%arg4, %c0] (%results[%results_3] [%c32] [%c1]) {id = 4 : i32} : (memref<64xf32, 1>)
        scf.reduce(%4 : !air.async.token) {
        ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
          %5 = air.wait_all async [%arg5, %arg6] 
          scf.reduce.return %5 : !air.async.token
        }
      }
      %3 = air.herd @herd_0 async [%async_token]  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c2_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
        %c0_2 = arith.constant 0 : index
        %4 = air.wait_all async 
        %async_token_3, %results_4 = air.execute -> (memref<32xf32, 2>) {
          %alloc = memref.alloc() : memref<32xf32, 2>
          air.execute_terminator %alloc : memref<32xf32, 2>
        }
        %5 = air.channel.get async [%4, %async_token_3]  @channel_1[%arg5, %c0_2] (%results_4[] [] []) {id = 6 : i32} : (memref<32xf32, 2>)
        %async_token_5 = air.execute [%5] {
          memref.dealloc %results_4 : memref<32xf32, 2>
        }
        air.herd_terminator
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<64xf32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// Tile / memtile DMA repeat count support.
// CHECK: aie.device(xcve2802)
// CHECK: %[[tileDMA_0_4:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 32)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 2>, 0, 8192)
// CHECK: %[[tileDMA_0_3:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 32)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 2>, 0, 8192)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb3, repeat_count = 32)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 1>, 0, 8192)
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2, repeat_count = 32)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 1>, 0, 8192)
// CHECK: @func10
#map = affine_map<()[s0] -> (s0 * 32)>
air.channel @channel_1 [2, 1]
func.func @func10(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg2) in (%arg3=%c2) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 3 : i64, y_size = 2 : i64} {
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<32x256xi32, 1>) {
        %alloc = memref.alloc() : memref<32x256xi32, 1>
        air.execute_terminator %alloc : memref<32x256xi32, 1>
      }
      %2 = scf.parallel (%arg4) = (%c0) to (%c2_0) step (%c1) init (%async_token) -> !air.async.token {
        %4 = air.channel.put async [%async_token]  @channel_1[%arg4, %c0] (%results[%c0, %c0, %c0] [%c32, %c32, %c256] [%c0, %c256, %c1]) {id = 4 : i32} : (memref<32x256xi32, 1>)
        scf.reduce(%4 : !air.async.token) {
        ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
          %5 = air.wait_all async [%arg5, %arg6] 
          scf.reduce.return %5 : !air.async.token
        }
      }
      %3 = air.herd @herd_0 async [%async_token]  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c2_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
        %c0_2 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        %c32_3 = arith.constant 32 : index
        %c256_5 = arith.constant 256 : index
        %4 = air.wait_all async 
        %async_token_3, %results_4 = air.execute -> (memref<32x256xi32, 2>) {
          %alloc = memref.alloc() : memref<32x256xi32, 2>
          air.execute_terminator %alloc : memref<32x256xi32, 2>
        }
        %5 = air.channel.get async [%4, %async_token_3]  @channel_1[%arg5, %c0_2] (%results_4[%c0_2, %c0_2, %c0_2] [%c32_3, %c32_3, %c256_5] [%c0_2, %c256_5, %c1_4]) {id = 6 : i32} : (memref<32x256xi32, 2>)
        %async_token_5 = air.execute [%5] {
          memref.dealloc %results_4 : memref<32x256xi32, 2>
        }
        air.herd_terminator
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<32x256xi32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// Bf16 datatype support.
// CHECK: aie.device(xcve2802)
// CHECK: %[[tileDMA_0_4:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xbf16, 2>, 0, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK: %[[tileDMA_0_3:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xbf16, 2>, 0, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb3, repeat_count = 1)
// CHECK:   memref<32x256xbf16, 1>, 0, 8192, [<size = 1, stride = 32>, <size = 32, stride = 256>, <size = 256, stride = 1>])
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2, repeat_count = 1)
// CHECK:   memref<32x256xbf16, 1>, 0, 8192, [<size = 1, stride = 32>, <size = 32, stride = 256>, <size = 256, stride = 1>])
// CHECK: @func11
#map = affine_map<()[s0] -> (s0 * 32)>
air.channel @channel_1 [2, 1]
func.func @func11(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg2) in (%arg3=%c2) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 3 : i64, y_size = 2 : i64} {
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %async_token, %results = air.execute -> (memref<32x256xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x256xbf16, 1>
        air.execute_terminator %alloc : memref<32x256xbf16, 1>
      }
      %2 = scf.parallel (%arg4) = (%c0) to (%c2_0) step (%c1) init (%async_token) -> !air.async.token {
        %4 = air.channel.put async [%async_token]  @channel_1[%arg4, %c0] (%results[%c0, %c0, %c0] [%c1, %c32, %c256] [%c32, %c256, %c1]) {id = 4 : i32} : (memref<32x256xbf16, 1>)
        scf.reduce(%4 : !air.async.token) {
        ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
          %5 = air.wait_all async [%arg5, %arg6] 
          scf.reduce.return %5 : !air.async.token
        }
      }
      %3 = air.herd @herd_0 async [%async_token]  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c2_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
        %c0_2 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        %c32_3 = arith.constant 32 : index
        %c256_5 = arith.constant 256 : index
        %c8_6 = arith.constant 8 : index
        %4 = air.wait_all async 
        %async_token_3, %results_4 = air.execute -> (memref<32x256xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x256xbf16, 2>
          air.execute_terminator %alloc : memref<32x256xbf16, 2>
        }
        %5 = air.channel.get async [%4, %async_token_3]  @channel_1[%arg5, %c0_2] (%results_4[%c0_2, %c0_2, %c0_2] [%c8_6, %c32_3, %c32_3] [%c32_3, %c256_5, %c1_4]) {id = 6 : i32} : (memref<32x256xbf16, 2>)
        %async_token_5 = air.execute [%5] {
          memref.dealloc %results_4 : memref<32x256xbf16, 2>
        }
        air.herd_terminator
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<32x256xbf16, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// 4x4 herd support.
// CHECK: aie.device(xcve2802)
// CHECK: %[[tile_2_0:.*]] = aie.tile(2, 0)
// CHECK: %[[tile_3_0:.*]] = aie.tile(3, 0)
// CHECK: %[[tile_2_1:.*]] = aie.tile(2, 1)
// CHECK: %[[tile_3_1:.*]] = aie.tile(3, 1)
// CHECK: %[[tile_4_1:.*]] = aie.tile(4, 1)
// CHECK: %[[tile_5_1:.*]] = aie.tile(5, 1)
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK: %[[tile_1_3:.*]] = aie.tile(1, 3)
// CHECK: %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK: %[[tile_3_3:.*]] = aie.tile(3, 3)
// CHECK: %[[tile_0_4:.*]] = aie.tile(0, 4)
// CHECK: %[[tile_1_4:.*]] = aie.tile(1, 4)
// CHECK: %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK: %[[tile_3_4:.*]] = aie.tile(3, 4)
// CHECK: %[[tile_0_5:.*]] = aie.tile(0, 5)
// CHECK: %[[tile_1_5:.*]] = aie.tile(1, 5)
// CHECK: %[[tile_2_5:.*]] = aie.tile(2, 5)
// CHECK: %[[tile_3_5:.*]] = aie.tile(3, 5)
// CHECK: %[[tile_0_6:.*]] = aie.tile(0, 6)
// CHECK: %[[tile_1_6:.*]] = aie.tile(1, 6)
// CHECK: %[[tile_2_6:.*]] = aie.tile(2, 6)
// CHECK: %[[tile_3_6:.*]] = aie.tile(3, 6)
// CHECK: %[[buf19:.*]] = aie.buffer(%[[tile_2_1]]) {{{.*}}} : memref<64x256xbf16, 1>
// CHECK: %[[buf18:.*]] = aie.buffer(%[[tile_3_1]]) {{{.*}}} : memref<64x256xbf16, 1>
// CHECK: %[[buf17:.*]] = aie.buffer(%[[tile_4_1]]) {{{.*}}} : memref<64x256xbf16, 1>
// CHECK: %[[buf16:.*]] = aie.buffer(%[[tile_5_1]]) {{{.*}}} : memref<64x256xbf16, 1>
// CHECK: %[[buf15:.*]] = aie.buffer(%[[tile_3_6]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf14:.*]] = aie.buffer(%[[tile_2_6]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf13:.*]] = aie.buffer(%[[tile_1_6]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf12:.*]] = aie.buffer(%[[tile_0_6]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf11:.*]] = aie.buffer(%[[tile_3_5]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf10:.*]] = aie.buffer(%[[tile_2_5]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf9:.*]] = aie.buffer(%[[tile_1_5]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf8:.*]] = aie.buffer(%[[tile_0_5]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf7:.*]] = aie.buffer(%[[tile_3_4]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf6:.*]] = aie.buffer(%[[tile_2_4]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf5:.*]] = aie.buffer(%[[tile_1_4]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf4:.*]] = aie.buffer(%[[tile_0_4]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf3:.*]] = aie.buffer(%[[tile_3_3]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf2:.*]] = aie.buffer(%[[tile_2_3]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf1:.*]] = aie.buffer(%[[tile_1_3]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: %[[buf0:.*]] = aie.buffer(%[[tile_0_3]]) {{{.*}}} : memref<16x16x4x4xbf16, 2>
// CHECK: aie.core(%[[tile_3_6]])
// CHECK: aie.core(%[[tile_2_6]])
// CHECK: aie.core(%[[tile_1_6]])
// CHECK: aie.core(%[[tile_0_6]])
// CHECK: aie.core(%[[tile_3_5]])
// CHECK: aie.core(%[[tile_2_5]])
// CHECK: aie.core(%[[tile_1_5]])
// CHECK: aie.core(%[[tile_0_5]])
// CHECK: aie.core(%[[tile_3_4]])
// CHECK: aie.core(%[[tile_2_4]])
// CHECK: aie.core(%[[tile_1_4]])
// CHECK: aie.core(%[[tile_0_4]])
// CHECK: aie.core(%[[tile_3_3]])
// CHECK: aie.core(%[[tile_2_3]])
// CHECK: aie.core(%[[tile_1_3]])
// CHECK: aie.core(%[[tile_0_3]])
// CHECK: aie.flow(%[[tile_5_1]], DMA : 0, %[[tile_2_0]], DMA : 0)
// CHECK: aie.flow(%[[tile_4_1]], DMA : 0, %[[tile_2_0]], DMA : 1)
// CHECK: aie.flow(%[[tile_3_1]], DMA : 0, %[[tile_3_0]], DMA : 0)
// CHECK: aie.flow(%[[tile_2_1]], DMA : 0, %[[tile_3_0]], DMA : 1)
// CHECK: aie.flow(%[[tile_0_3]], DMA : 0, %[[tile_5_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_1_3]], DMA : 0, %[[tile_4_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_2_3]], DMA : 0, %[[tile_3_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_3_3]], DMA : 0, %[[tile_2_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_0_4]], DMA : 0, %[[tile_5_1]], DMA : 1)
// CHECK: aie.flow(%[[tile_1_4]], DMA : 0, %[[tile_4_1]], DMA : 1)
// CHECK: aie.flow(%[[tile_2_4]], DMA : 0, %[[tile_3_1]], DMA : 1)
// CHECK: aie.flow(%[[tile_3_4]], DMA : 0, %[[tile_2_1]], DMA : 1)
// CHECK: aie.flow(%[[tile_0_5]], DMA : 0, %[[tile_5_1]], DMA : 2)
// CHECK: aie.flow(%[[tile_1_5]], DMA : 0, %[[tile_4_1]], DMA : 2)
// CHECK: aie.flow(%[[tile_2_5]], DMA : 0, %[[tile_3_1]], DMA : 2)
// CHECK: aie.flow(%[[tile_3_5]], DMA : 0, %[[tile_2_1]], DMA : 2)
// CHECK: aie.flow(%[[tile_0_6]], DMA : 0, %[[tile_5_1]], DMA : 3)
// CHECK: aie.flow(%[[tile_1_6]], DMA : 0, %[[tile_4_1]], DMA : 3)
// CHECK: aie.flow(%[[tile_2_6]], DMA : 0, %[[tile_3_1]], DMA : 3)
// CHECK: aie.flow(%[[tile_3_6]], DMA : 0, %[[tile_2_1]], DMA : 3)
// CHECK: aie.memtile_dma(%[[tile_5_1]])
// CHECK: aie.memtile_dma(%[[tile_4_1]])
// CHECK: aie.memtile_dma(%[[tile_3_1]])
// CHECK: aie.memtile_dma(%[[tile_2_1]])
// CHECK: @func12

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 256 + 64)>
#map2 = affine_map<()[s0] -> (s0 * 256 + 128)>
#map3 = affine_map<()[s0] -> (s0 * 256 + 192)>
#map4 = affine_map<()[s0] -> (s0 * 64)>
module {
  air.channel @channel_12 [4, 1]
  air.channel @channel_10 [4, 4]
  func.func @func12(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg2) : memref<512x512xbf16> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %async_token, %results = air.execute -> (index) {
        %11 = affine.apply #map()[%arg4]
        air.execute_terminator %11 : index
      }
      %1 = affine.apply #map()[%arg3]
      %2 = air.channel.get async [%async_token]  @channel_12[%c0, %c0] (%arg7[%1, %results] [%c64, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
      %3 = affine.apply #map1()[%arg3]
      %4 = air.channel.get async [%async_token]  @channel_12[%c1, %c0] (%arg7[%3, %results] [%c64, %c256] [%c512, %c1]) {id = 4 : i32} : (memref<512x512xbf16>)
      %5 = affine.apply #map2()[%arg3]
      %6 = air.channel.get async [%async_token]  @channel_12[%c2_0, %c0] (%arg7[%5, %results] [%c64, %c256] [%c512, %c1]) {id = 5 : i32} : (memref<512x512xbf16>)
      %7 = affine.apply #map3()[%arg3]
      %8 = air.channel.get async [%async_token]  @channel_12[%c3, %c0] (%arg7[%7, %results] [%c64, %c256] [%c512, %c1]) {id = 6 : i32} : (memref<512x512xbf16>)
      %9 = air.wait_all async [%2, %4, %6, %8] 
      %10 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 3 : i64, y_size = 4 : i64} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c192 = arith.constant 192 : index
        %c64_3 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c1_4 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_5 = arith.constant 0 : index
        %c256_6 = arith.constant 256 : index
        %async_token_7, %results_8 = air.execute -> (memref<256x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<256x256xbf16, 1>
          air.execute_terminator %alloc : memref<256x256xbf16, 1>
        }
        %11 = scf.parallel (%arg8, %arg9) = (%c0_5, %c0_5) to (%c4, %c4) step (%c1_4, %c1_4) init (%async_token_7) -> !air.async.token {
          %async_token_10, %results_11 = air.execute -> (index) {
            %19 = affine.apply #map4()[%arg8]
            air.execute_terminator %19 : index
          }
          %async_token_12, %results_13 = air.execute -> (index) {
            %19 = affine.apply #map4()[%arg9]
            air.execute_terminator %19 : index
          }
          %18 = air.channel.get async [%async_token_7, %async_token_12, %async_token_10]  @channel_10[%arg8, %arg9] (%results_8[%results_11, %results_13] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<256x256xbf16, 1>)
          scf.reduce(%18 : !air.async.token) {
          ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
            %19 = air.wait_all async [%arg10, %arg11] 
            scf.reduce.return %19 : !air.async.token
          }
        }
        %12 = air.herd @herd_0 async [%async_token_7]  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, link_with = "mm.o", x_loc = 0 : i64, y_loc = 3 : i64} {
          %c64_10 = arith.constant 64 : index
          %c256_11 = arith.constant 256 : index
          %c4_12 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c1_13 = arith.constant 1 : index
          %c0_14 = arith.constant 0 : index
          %async_token_15, %results_16 = air.execute -> (memref<16x16x4x4xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x16x4x4xbf16, 2>
            air.execute_terminator %alloc : memref<16x16x4x4xbf16, 2>
          }
          %18 = air.channel.put async [%async_token_15]  @channel_10[%arg8, %arg9] (%results_16[%c0_14, %c0_14, %c0_14] [%c64_10, %c16, %c4_12] [%c4_12, %c256_11, %c1_13]) {id = 44 : i32} : (memref<16x16x4x4xbf16, 2>)
          %async_token_17 = air.execute [%18] {
            memref.dealloc %results_16 : memref<16x16x4x4xbf16, 2>
          }
          air.herd_terminator
        }
        %13 = air.channel.put async [%12]  @channel_12[%c0_5, %c0_5] (%results_8[%c0_5, %c0_5] [%c64_3, %c256_6] [%c256_6, %c1_4]) {id = 45 : i32} : (memref<256x256xbf16, 1>)
        %14 = air.channel.put async [%12]  @channel_12[%c1_4, %c0_5] (%results_8[%c64_3, %c0_5] [%c64_3, %c256_6] [%c256_6, %c1_4]) {id = 46 : i32} : (memref<256x256xbf16, 1>)
        %15 = air.channel.put async [%12]  @channel_12[%c2_2, %c0_5] (%results_8[%c128, %c0_5] [%c64_3, %c256_6] [%c256_6, %c1_4]) {id = 47 : i32} : (memref<256x256xbf16, 1>)
        %16 = air.channel.put async [%12]  @channel_12[%c3_1, %c0_5] (%results_8[%c192, %c0_5] [%c64_3, %c256_6] [%c256_6, %c1_4]) {id = 48 : i32} : (memref<256x256xbf16, 1>)
        %17 = air.wait_all async [%13, %14, %15, %16] 
        %async_token_9 = air.execute [%17] {
          memref.dealloc %results_8 : memref<256x256xbf16, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

