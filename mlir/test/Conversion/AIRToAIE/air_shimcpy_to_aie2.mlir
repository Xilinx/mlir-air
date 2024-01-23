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
// CHECK:    %[[VAL_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
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
func.func @func1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
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
// CHECK:   %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb2)
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
// CHECK:   aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%[[VAL_2]], Release, 1)
// CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:   aie.end
// CHECK: aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK: aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId2(S2MM, 0, 2)
// CHECK: memref.global "public" @airMemcpyId2 : memref<512xi32, 2>
// CHECK: aie.shim_dma_allocation @airMemcpyId1(MM2S, 0, 2)
// CHECK: memref.global "public" @airMemcpyId1 : memref<1024xi32, 2>
func.func @func1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
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
// CHECK:         %[[VAL_6:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(MM2S, 0, ^bb4, ^bb2)
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
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
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
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
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
// CHECK:         %[[VAL_21:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_22:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_23:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = {{.*}}} : memref<1024xi32, 2>

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
// CHECK:           aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_17]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:         aie.flow(%[[VAL_2]], DMA : 1, %[[VAL_4]], DMA : 0)

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
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 2)
// CHECK: memref.global "public" @airMemcpyId7 : memref<1024xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 2)
// CHECK: memref.global "public" @airMemcpyId2 : memref<1024xi32, 1>

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
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="func4"} {
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
    %token_5 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="func5"} {
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
// CHECK: %[[tile_2_1:.*]] = aie.tile(2, 1)
// CHECK: %[[tile_3_1:.*]] = aie.tile(3, 1)
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
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<4x4xi32, 1>, 0, 16)
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:  // pred: ^bb5
// CHECK:   aie.dma_start(S2MM, 1, ^bb4, ^bb2)
// CHECK: ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<8x16xi32, 1>, 0, 16, [<size = 4, stride = 16>, <size = 4, stride = 1>])
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:  // pred: ^bb0
// CHECK:   aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK: ^bb6:  // 2 preds: ^bb5, ^bb6
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<4x4xi32, 1>, 0, 16)
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb6

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
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<64x256xi32, 1>, 32768, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:  // pred: ^bb0
// CHECK:   aie.end

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
