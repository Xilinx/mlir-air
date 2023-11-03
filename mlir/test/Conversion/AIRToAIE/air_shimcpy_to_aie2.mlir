//===- air_shimcpy_to_aie2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:  %[[VAL_0:.*]] = AIE.tile(2, 3)
// CHECK:  %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:  %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK:  %[[VAL_5:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:    %[[VAL_6:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    AIE.useLock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:    AIE.dmaBd(<%[[VAL_4]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:    AIE.nextBd ^bb1
// CHECK:  ^bb2:
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %[[VAL_7:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:    AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:    AIE.end
// CHECK:  AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
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

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK: %[[VAL_0:.*]] = AIE.tile(2, 3)
// CHECK: %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK: %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 1 : i32}
// CHECK: %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32}
// CHECK: %[[VAL_4:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK: %[[VAL_5:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK: %[[VAL_6:.*]] = AIE.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK: %[[VAL_7:.*]] = AIE.buffer(%[[VAL_0]]) {{.*}} : memref<512xi32, 2>
// CHECK: %[[VAL_8:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:   %[[VAL_9:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb2:
// CHECK:   AIE.end
// CHECK: ^bb3:
// CHECK:   %[[VAL_10:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_7]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:   AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: }
// CHECK: %[[VAL_11:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:   AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:   AIE.end
// CHECK: AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK: AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
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
// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_1]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_5:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_6:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_7:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_1]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_7]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_1]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:           AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:           AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)

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
// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 1)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_4:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_5:.*]] = AIE.lock(%[[VAL_2]], 7) {init = 1 : i32}
// CHECK:         %[[VAL_6:.*]] = AIE.lock(%[[VAL_2]], 6) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = AIE.lock(%[[VAL_2]], 5) {init = 1 : i32}
// CHECK:         %[[VAL_8:.*]] = AIE.lock(%[[VAL_2]], 4) {init = 0 : i32}
// CHECK:         %[[VAL_9:.*]] = AIE.lock(%[[VAL_2]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_10:.*]] = AIE.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_11:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_12:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_17:.*]] = AIE.lock(%[[VAL_3]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_18:.*]] = AIE.lock(%[[VAL_3]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_19:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_20:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_21:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_22:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<1024xi32, 1>
// CHECK:         %[[VAL_23:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = {{.*}}} : memref<1024xi32, 2>

// CHECK:    AIE.mem(%[[VAL_3]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_18]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_3]]) {
// CHECK:           AIE.useLock(%[[VAL_18]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_20]], Release, 1)
// CHECK:           AIE.useLock(%[[VAL_17]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:         AIE.flow(%[[VAL_2]], DMA : 1, %[[VAL_4]], DMA : 0)

// CHECK: AIE.memTileDMA(%[[VAL_2]]) {
// CHECK:   AIE.dmaStart(S2MM, 0, ^bb1, ^bb7)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_12]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb2:
// CHECK:   AIE.end
// CHECK: ^bb3:
// CHECK:   AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: ^bb5:
// CHECK:   AIE.dmaStart(MM2S, 0, ^bb6, ^bb3)
// CHECK: ^bb6:
// CHECK:   AIE.useLock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:   AIE.nextBd ^bb6
// CHECK: ^bb7:
// CHECK:   AIE.dmaStart(MM2S, 1, ^bb8, ^bb5)
// CHECK: ^bb8:
// CHECK:   AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:   AIE.nextBd ^bb8
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
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(3, 3)
// CHECK:         %[[VAL_4:.*]] = AIE.tile(4, 3)
// CHECK:         %[[VAL_5:.*]] = AIE.tile(5, 3)
// CHECK:         %[[VAL_6:.*]] = AIE.tile(2, 4)
// CHECK:         %[[VAL_7:.*]] = AIE.tile(3, 4)
// CHECK:         %[[VAL_8:.*]] = AIE.tile(4, 4)
// CHECK:         %[[VAL_9:.*]] = AIE.tile(5, 4)
// CHECK:         %[[VAL_10:.*]] = AIE.tile(2, 5)
// CHECK:         %[[VAL_11:.*]] = AIE.tile(3, 5)
// CHECK:         %[[VAL_12:.*]] = AIE.tile(4, 5)
// CHECK:         %[[VAL_13:.*]] = AIE.tile(5, 5)
// CHECK:         %[[VAL_14:.*]] = AIE.tile(2, 6)
// CHECK:         %[[VAL_15:.*]] = AIE.tile(3, 6)
// CHECK:         %[[VAL_16:.*]] = AIE.tile(4, 6)
// CHECK:         %[[VAL_17:.*]] = AIE.tile(5, 6)

// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_10]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_14]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_2]], DMA : 1)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_4]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_5]], DMA : 0)

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
air.channel @channel_6 [1, 1] {broadcast_shape = [1, 4]}
air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
air.channel @channel_8 [1, 1]
func.func @func5(%arg0 : memref<1024xi32>) -> () {
  %token_0 = air.channel.put async @channel_8[] (%arg0[] [] []) {id = 3 : i32} : (memref<1024xi32>)
  %token_10 = air.segment @segment0 async {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : index
    %c512 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
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
