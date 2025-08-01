//===- air_shimcpy_to_aie2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" --split-input-file | FileCheck %s
// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1 use-lock-race-condition-fix=true" --split-input-file | FileCheck %s  --check-prefix=RACECONDFIX

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(0, 2)
// CHECK:  %[[VAL_1:.*]] = aie.tile(0, 0)
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
// CHECK:  aie.shim_dma_allocation @airMemcpyId1(MM2S, 0, 0)
// CHECK: @func1
// RACECONDFIX: @func1
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

// CHECK-LABEL:   aie.device(npu1) {
// CHECK: %[[VAL_0:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL_1:.*]] = aie.tile(0, 0)
// CHECK: %[[VAL_2:.*]] = aie.lock(%[[VAL_0]], 3) {init = 1 : i32}
// CHECK: %[[VAL_3:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32}
// CHECK: %[[VAL_4:.*]] = aie.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK: %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK: %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2>
// CHECK: %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<512xi32, 2>
// CHECK: %[[VAL_8:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:   %[[VAL_9:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_7]] : memref<512xi32, 2>, 0, 512)
// CHECK:   aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   %[[VAL_10:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_3]], Release, 1)
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
// CHECK: aie.shim_dma_allocation @airMemcpyId2(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<512xi32, 2>
// CHECK: aie.shim_dma_allocation @airMemcpyId1(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId1 : memref<1024xi32, 2>
// CHECK: @func2
// RACECONDFIX: @func2
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
// CHECK-LABEL:   aie.device(npu1) {
// CHECK:         %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:         %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:         %[[VAL_2:.*]] = aie.lock(%[[VAL_1]], 3) {init = 1 : i32}
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_6:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_7:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<512xi32, 2>

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_7]] : memref<512xi32, 2>, 0, 512)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(S2MM, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
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
// CHECK:         aie.shim_dma_allocation @air_channel_1(S2MM, 0, 0)
// CHECK:         memref.global "public" @air_channel_1 : memref<512xi32, 2>
// CHECK:         aie.shim_dma_allocation @air_channel_0(MM2S, 0, 0)
// CHECK:         memref.global "public" @air_channel_0 : memref<1024xi32, 2>
// CHECK: @func3
// RACECONDFIX: @func3
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
// CHECK-LABEL:   aie.device(npu1) {
// CHECK:         %[[VAL_2:.*]] = aie.tile(0, 1)
// CHECK:         %[[VAL_3:.*]] = aie.tile(0, 2)
// CHECK:         %[[VAL_4:.*]] = aie.tile(0, 0)
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
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(S2MM, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_23]] : memref<1024xi32, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_18]], Release, 1)
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

// CHECK: aie.memtile_dma(%[[VAL_2]]) {
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb7)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:
// CHECK:   aie.end
// CHECK: ^bb3:
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:
// CHECK:   aie.dma_start(S2MM, 0, ^bb6, ^bb3)
// CHECK: ^bb6:
// CHECK:   aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_21]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: ^bb7:
// CHECK:   aie.dma_start(S2MM, 1, ^bb8, ^bb5)
// CHECK: ^bb8:
// CHECK:   aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[VAL_22]] : memref<1024xi32, 1>, 0, 1024)
// CHECK:   aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:   aie.next_bd ^bb8
// CHECK: }
// CHECK: aie.shim_dma_allocation @air_channel_5(S2MM, 0, 0)
// CHECK: memref.global "public" @air_channel_5 : memref<1024xi32, 1>
// CHECK: aie.shim_dma_allocation @air_channel_2(MM2S, 0, 0)
// CHECK: memref.global "public" @air_channel_2 : memref<1024xi32, 1>
// CHECK: @func4
// RACECONDFIX: @func4
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

// -----

// L2 to L1 broadcast
// CHECK: aie.device
// CHECK:         %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:         %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:         %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:         %[[VAL_3:.*]] = aie.tile(1, 2)
// CHECK:         %[[VAL_4:.*]] = aie.tile(2, 2)
// CHECK:         %[[VAL_5:.*]] = aie.tile(3, 2)
// CHECK:         %[[VAL_6:.*]] = aie.tile(0, 3)
// CHECK:         %[[VAL_7:.*]] = aie.tile(1, 3)
// CHECK:         %[[VAL_8:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_9:.*]] = aie.tile(3, 3)
// CHECK:         %[[VAL_10:.*]] = aie.tile(0, 4)
// CHECK:         %[[VAL_11:.*]] = aie.tile(1, 4)
// CHECK:         %[[VAL_12:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_13:.*]] = aie.tile(3, 4)
// CHECK:         %[[VAL_14:.*]] = aie.tile(0, 5)
// CHECK:         %[[VAL_15:.*]] = aie.tile(1, 5)
// CHECK:         %[[VAL_16:.*]] = aie.tile(2, 5)
// CHECK:         %[[VAL_17:.*]] = aie.tile(3, 5)

// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_10]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_14]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_2]], DMA : 1)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_4]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_5]], DMA : 0)

// CHECK:         aie.shim_dma_allocation @air_channel_8(MM2S, 0, 0)
// CHECK:         memref.global "public" @air_channel_8 : memref<1024xi32, 1>
// CHECK: @func5

// RACECONDFIX: aie.device
// RACECONDFIX:   aie.memtile_dma(%{{.*}}) {
// RACECONDFIX:     %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// RACECONDFIX:   ^bb1:
// RACECONDFIX:     aie.use_lock(%[[lock_0_1_2:.*]], AcquireGreaterEqual, 1)
// RACECONDFIX:     aie.dma_bd(%[[buf32:.*]] : memref<1024xi32, 1>, 0, 1024)
// RACECONDFIX:     aie.use_lock(%[[lock_0_1_1:.*]], Release, 1)
// RACECONDFIX:     aie.next_bd ^bb1
// RACECONDFIX:   ^bb2:
// RACECONDFIX:     aie.end
// RACECONDFIX:   ^bb3:
// RACECONDFIX:     %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb2)
// RACECONDFIX:   ^bb4:
// RACECONDFIX:     aie.use_lock(%[[lock_0_1_0:.*]], AcquireGreaterEqual, 1)
// RACECONDFIX:     aie.dma_bd(%[[buf32]] : memref<1024xi32, 1>, 0, 512)
// RACECONDFIX:     aie.use_lock(%[[lock_0_1:.*]], Release, 1)
// RACECONDFIX:     aie.next_bd ^bb4
// RACECONDFIX:   ^bb5:
// RACECONDFIX:     %2 = aie.dma_start(S2MM, 0, ^bb6, ^bb3)
// RACECONDFIX:   ^bb6:
// RACECONDFIX:     aie.use_lock(%[[lock_0_1_1]], AcquireGreaterEqual, 1)
// RACECONDFIX:     aie.dma_bd(%[[buf32]] : memref<1024xi32, 1>, 0, 1024)
// RACECONDFIX:     aie.use_lock(%[[lock_0_1_2]], Release, 1)
// RACECONDFIX:     aie.next_bd ^bb7
// RACECONDFIX:   ^bb7:
// RACECONDFIX:     aie.use_lock(%[[lock_0_1]], AcquireGreaterEqual, 1)
// RACECONDFIX:     aie.dma_bd(%[[buf32]] : memref<1024xi32, 1>, 0, 0)
// RACECONDFIX:     aie.use_lock(%[[lock_0_1_0]], Release, 1)
// RACECONDFIX:     aie.next_bd ^bb6
// RACECONDFIX: @func5

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
    }
  }
  return
}

// -----

// L3 to L1 parallel shim dmas
// CHECK: aie.device(npu1)
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK: %[[tile_1_3:.*]] = aie.tile(1, 3)
// CHECK: %[[tile_0_4:.*]] = aie.tile(0, 4)
// CHECK: %[[tile_1_4:.*]] = aie.tile(1, 4)

// CHECK:  aie.flow(%[[tile_0_3]], DMA : 0, %[[tile_0_0]], DMA : 0)
// CHECK:  aie.flow(%[[tile_0_4]], DMA : 0, %[[tile_0_0]], DMA : 1)
// CHECK:  aie.flow(%[[tile_1_3]], DMA : 0, %[[tile_1_0]], DMA : 0)
// CHECK:  aie.flow(%[[tile_1_4]], DMA : 0, %[[tile_1_0]], DMA : 1)
// CHECK:  aie.shim_dma_allocation @air_channel_0_0(S2MM, 0, 0)
// CHECK:  memref.global "public" @air_channel_0_0 : memref<4x4xi32, 2>
// CHECK:  aie.shim_dma_allocation @air_channel_0_1(S2MM, 1, 0)
// CHECK:  memref.global "public" @air_channel_0_1 : memref<4x4xi32, 2>
// CHECK:  aie.shim_dma_allocation @air_channel_0_2(S2MM, 0, 1)
// CHECK:  memref.global "public" @air_channel_0_2 : memref<4x4xi32, 2>
// CHECK:  aie.shim_dma_allocation @air_channel_0_3(S2MM, 1, 1)
// CHECK:  memref.global "public" @air_channel_0_3 : memref<4x4xi32, 2>

// CHECK: @func6
// CHECK: air.channel.get{{.*}}metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<8x8xi32>)
// RACECONDFIX: @func6
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
      }
    }
  }
  return
}

// -----

// Multi-dimensional memref copy to wraps and strides
// CHECK: aie.device(npu1)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<4x4xi32, 1>, 0, 16)
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:  // pred: ^bb5
// CHECK:   aie.dma_start(S2MM, 0, ^bb4, ^bb2)
// CHECK: ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<4x4xi32, 1>, 0, 16)
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb5:  // pred: ^bb0
// CHECK:   aie.dma_start(S2MM, 1, ^bb6, ^bb3)
// CHECK: ^bb6:  // 2 preds: ^bb5, ^bb6
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<8x16xi32, 1>, 0, 16, [<size = 4, stride = 16>, <size = 4, stride = 1>])
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb6
// CHECK: @func7
// RACECONDFIX: @func7
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
    }
    %buf0 = memref.alloc() : memref<4x4xi32, 1>
    %buf1 = memref.alloc() : memref<8x16xi32, 1>
    air.channel.get @channel_0[] (%buf0[] [] []) {id = 2 : i32} : (memref<4x4xi32, 1>)
    air.channel.put @channel_1[] (%buf0[] [] []) {id = 3 : i32} : (memref<4x4xi32, 1>)
    air.channel.get @channel_2[] (%buf1[%c0, %c0] [%c4, %c4] [%c16, %c1]) {id = 4 : i32} : (memref<8x16xi32, 1>)
    memref.dealloc %buf0 : memref<4x4xi32, 1>
    memref.dealloc %buf1 : memref<8x16xi32, 1>
  }
  air.channel.get @channel_1[] (%arg1[] [] []) {id = 4 : i32} : (memref<16x8xi32>)
  return
}

// -----

// Multi-dimensional memref copy to wraps and strides, with offsets having more dims than memref type.
// CHECK: aie.device(npu1)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:   aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd({{.*}} : memref<64x256xi32, 1>, 8192, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK:   aie.use_lock({{.*}}, Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb2:  // pred: ^bb0
// CHECK:   aie.end
// CHECK: @func8
// RACECONDFIX: @func8
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
    }
    %buf0 = memref.alloc() : memref<64x256xi32, 1>
    air.channel.put @channel_0[] (%buf0[%c0, %c32, %c0] [%c8, %c32, %c32] [%c32, %c256, %c1]) : (memref<64x256xi32, 1>)
    memref.dealloc %buf0 : memref<64x256xi32, 1>
  }
  return
}
// -----

// 1D scf.parallel iteration space support.
// CHECK: aie.device(npu1)
// CHECK: %[[tileDMA_0_4:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32xf32, 2>, 0, 32)
// CHECK: %[[tileDMA_0_3:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32xf32, 2>, 0, 32)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:   aie.dma_bd({{.*}} : memref<64xf32, 1>, 0, 32)
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<64xf32, 1>, 32, 32)
// CHECK: @func9
// RACECONDFIX: @func9
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
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<64xf32, 1>
      }
    }
  }
  return
}

// -----

// Tile / memtile DMA repeat count support.
// CHECK: aie.device(npu1)
// CHECK: %[[tileDMA_0_4:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 2>, 0, 8192)
// CHECK: %[[tileDMA_0_3:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 2>, 0, 8192)
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 1>, 0, 8192)
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xi32, 1>, 0, 8192)
// CHECK: @func10
// RACECONDFIX: @func10
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
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<32x256xi32, 1>
      }
    }
  }
  return
}

// -----

// Bf16 datatype support.
// CHECK: aie.device(npu1)
// CHECK: %[[tileDMA_0_4:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xbf16, 2>, 0, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK: %[[tileDMA_0_3:.*]] = aie.mem
// CHECK:   aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   aie.dma_bd({{.*}} : memref<32x256xbf16, 2>, 0, 8192, [<size = 8, stride = 32>, <size = 32, stride = 256>, <size = 32, stride = 1>])
// CHECK: %[[memTileDMA_2_1:.*]] = aie.memtile_dma
// CHECK:   aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:   memref<32x256xbf16, 1>, 0, 8192)
// CHECK:   aie.dma_start(MM2S, 1, ^bb4, ^bb2)
// CHECK:   memref<32x256xbf16, 1>, 0, 8192)
// CHECK: @func11
// RACECONDFIX: @func11
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
      }
      %async_token_1 = air.execute [%3] {
        memref.dealloc %results : memref<32x256xbf16, 1>
      }
    }
  }
  return
}

// -----

// 4x4 herd support.
// CHECK: aie.device(npu1)
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK: %[[tile_2_0:.*]] = aie.tile(2, 0)
// CHECK: %[[tile_3_0:.*]] = aie.tile(3, 0)
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK: %[[tile_1_1:.*]] = aie.tile(1, 1)
// CHECK: %[[tile_2_1:.*]] = aie.tile(2, 1)
// CHECK: %[[tile_3_1:.*]] = aie.tile(3, 1)
// CHECK: %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK: %[[tile_1_2:.*]] = aie.tile(1, 2)
// CHECK: %[[tile_2_2:.*]] = aie.tile(2, 2)
// CHECK: %[[tile_3_2:.*]] = aie.tile(3, 2)
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
// CHECK: %[[buf19:.*]] = aie.buffer(%[[tile_0_1]]) {sym_name = "buf19"} : memref<64x256xbf16, 1> 
// CHECK: %[[buf18:.*]] = aie.buffer(%[[tile_1_1]]) {sym_name = "buf18"} : memref<64x256xbf16, 1> 
// CHECK: %[[buf17:.*]] = aie.buffer(%[[tile_2_1]]) {sym_name = "buf17"} : memref<64x256xbf16, 1> 
// CHECK: %[[buf16:.*]] = aie.buffer(%[[tile_3_1]]) {sym_name = "buf16"} : memref<64x256xbf16, 1> 
// CHECK: %[[buf15:.*]] = aie.buffer(%[[tile_3_5]]) {sym_name = "buf15"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf14:.*]] = aie.buffer(%[[tile_2_5]]) {sym_name = "buf14"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf13:.*]] = aie.buffer(%[[tile_1_5]]) {sym_name = "buf13"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf12:.*]] = aie.buffer(%[[tile_0_5]]) {sym_name = "buf12"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf11:.*]] = aie.buffer(%[[tile_3_4]]) {sym_name = "buf11"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf10:.*]] = aie.buffer(%[[tile_2_4]]) {sym_name = "buf10"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf9:.*]] = aie.buffer(%[[tile_1_4]]) {sym_name = "buf9"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf8:.*]] = aie.buffer(%[[tile_0_4]]) {sym_name = "buf8"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf7:.*]] = aie.buffer(%[[tile_3_3]]) {sym_name = "buf7"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf6:.*]] = aie.buffer(%[[tile_2_3]]) {sym_name = "buf6"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf5:.*]] = aie.buffer(%[[tile_1_3]]) {sym_name = "buf5"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf4:.*]] = aie.buffer(%[[tile_0_3]]) {sym_name = "buf4"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf3:.*]] = aie.buffer(%[[tile_3_2]]) {sym_name = "buf3"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf2:.*]] = aie.buffer(%[[tile_2_2]]) {sym_name = "buf2"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf1:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "buf1"} : memref<16x16x4x4xbf16, 2> 
// CHECK: %[[buf0:.*]] = aie.buffer(%[[tile_0_2]]) {sym_name = "buf0"} : memref<16x16x4x4xbf16, 2> 
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
// CHECK: aie.core(%[[tile_3_2]])
// CHECK: aie.core(%[[tile_2_2]])
// CHECK: aie.core(%[[tile_1_2]])
// CHECK: aie.core(%[[tile_0_2]])
// CHECK: aie.flow(%[[tile_0_1]], DMA : 0, %[[tile_0_0]], DMA : 0)
// CHECK: aie.flow(%[[tile_1_1]], DMA : 0, %[[tile_1_0]], DMA : 0)
// CHECK: aie.flow(%[[tile_2_1]], DMA : 0, %[[tile_2_0]], DMA : 0)
// CHECK: aie.flow(%[[tile_3_1]], DMA : 0, %[[tile_3_0]], DMA : 0)
// CHECK: aie.flow(%[[tile_0_2]], DMA : 0, %[[tile_0_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_0_3]], DMA : 0, %[[tile_0_1]], DMA : 1)
// CHECK: aie.flow(%[[tile_0_4]], DMA : 0, %[[tile_0_1]], DMA : 2)
// CHECK: aie.flow(%[[tile_0_5]], DMA : 0, %[[tile_0_1]], DMA : 3)
// CHECK: aie.flow(%[[tile_1_2]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_1_3]], DMA : 0, %[[tile_1_1]], DMA : 1)
// CHECK: aie.flow(%[[tile_1_4]], DMA : 0, %[[tile_1_1]], DMA : 2)
// CHECK: aie.flow(%[[tile_1_5]], DMA : 0, %[[tile_1_1]], DMA : 3)
// CHECK: aie.flow(%[[tile_2_2]], DMA : 0, %[[tile_1_1]], DMA : 4)
// CHECK: aie.flow(%[[tile_2_3]], DMA : 0, %[[tile_1_1]], DMA : 5)
// CHECK: aie.flow(%[[tile_2_4]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_2_5]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_3_2]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_3_3]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_3_4]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.flow(%[[tile_3_5]], DMA : 0, %[[tile_1_1]], DMA : 0)
// CHECK: aie.memtile_dma(%[[tile_0_1]])
// CHECK: aie.memtile_dma(%[[tile_1_1]])
// CHECK: aie.memtile_dma(%[[tile_2_1]])
// CHECK: aie.memtile_dma(%[[tile_3_1]])
// CHECK: @func12

// RACECONDFIX: aie.device(npu1)
// RACECONDFIX: aie.memtile_dma(%[[mem_tile_0_1:.*]])
// RACECONDFIX-COUNT-3: aie.dma_bd(%[[buf19:.*]] : memref<64x256xbf16, 1>, 0, 0)
// RACECONDFIX: aie.dma_bd(%[[buf19]] : memref<64x256xbf16, 1>, 0, 16384)
// RACECONDFIX: aie.memtile_dma(%[[mem_tile_1_1:.*]])
// RACECONDFIX-COUNT-11: aie.dma_bd(%[[buf18:.*]] : memref<64x256xbf16, 1>, 0, 0)
// RACECONDFIX: aie.dma_bd(%[[buf18]] : memref<64x256xbf16, 1>, 0, 16384)
// RACECONDFIX: @func12

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 256 + 64)>
#map2 = affine_map<()[s0] -> (s0 * 256 + 128)>
#map3 = affine_map<()[s0] -> (s0 * 256 + 192)>
#map4 = affine_map<()[s0] -> (s0 * 64)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
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
      %2 = air.channel.get async [%async_token]  @channel_0[] (%arg7[%1, %results] [%c64, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
      %3 = affine.apply #map1()[%arg3]
      %4 = air.channel.get async [%async_token]  @channel_1[] (%arg7[%3, %results] [%c64, %c256] [%c512, %c1]) {id = 4 : i32} : (memref<512x512xbf16>)
      %5 = affine.apply #map2()[%arg3]
      %6 = air.channel.get async [%async_token]  @channel_2[] (%arg7[%5, %results] [%c64, %c256] [%c512, %c1]) {id = 5 : i32} : (memref<512x512xbf16>)
      %7 = affine.apply #map3()[%arg3]
      %8 = air.channel.get async [%async_token]  @channel_3[] (%arg7[%7, %results] [%c64, %c256] [%c512, %c1]) {id = 6 : i32} : (memref<512x512xbf16>)
      %9 = air.wait_all async [%2, %4, %6, %8] 
      %10 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c192 = arith.constant 192 : index
        %c64_3 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c1_4 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_5 = arith.constant 0 : index
        %c256_6 = arith.constant 256 : index
        %async_token_4, %results_5 = air.execute -> (memref<64x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1>
          air.execute_terminator %alloc : memref<64x256xbf16, 1>
        }
        %async_token_5, %results_6 = air.execute -> (memref<64x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1>
          air.execute_terminator %alloc : memref<64x256xbf16, 1>
        }
        %async_token_6, %results_7 = air.execute -> (memref<64x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1>
          air.execute_terminator %alloc : memref<64x256xbf16, 1>
        }
        %async_token_7, %results_8 = air.execute -> (memref<64x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x256xbf16, 1>
          air.execute_terminator %alloc : memref<64x256xbf16, 1>
        }
        %18 = air.channel.get async [%async_token_4]  @channel_10[%c0_5, %c0_5] (%results_5[%c0_5, %c0_5] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %20 = air.channel.get async [%async_token_4]  @channel_10[%c0_5, %c1_4] (%results_5[%c0_5, %c64_3] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %21 = air.channel.get async [%async_token_4]  @channel_10[%c0_5, %c2_2] (%results_5[%c0_5, %c128] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %22 = air.channel.get async [%async_token_4]  @channel_10[%c0_5, %c3_1] (%results_5[%c0_5, %c192] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %23 = air.channel.get async [%async_token_5]  @channel_10[%c1_4, %c0_5] (%results_6[%c0_5, %c0_5] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %24 = air.channel.get async [%async_token_5]  @channel_10[%c1_4, %c1_4] (%results_6[%c0_5, %c64_3] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %25 = air.channel.get async [%async_token_5]  @channel_10[%c1_4, %c2_2] (%results_6[%c0_5, %c128] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %26 = air.channel.get async [%async_token_5]  @channel_10[%c1_4, %c3_1] (%results_6[%c0_5, %c192] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %27 = air.channel.get async [%async_token_6]  @channel_10[%c2_2, %c0_5] (%results_6[%c0_5, %c0_5] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %28 = air.channel.get async [%async_token_6]  @channel_10[%c2_2, %c1_4] (%results_6[%c0_5, %c64_3] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %29 = air.channel.get async [%async_token_6]  @channel_10[%c2_2, %c2_2] (%results_6[%c0_5, %c128] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %30 = air.channel.get async [%async_token_6]  @channel_10[%c2_2, %c3_1] (%results_6[%c0_5, %c192] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %31 = air.channel.get async [%async_token_7]  @channel_10[%c3_1, %c0_5] (%results_6[%c0_5, %c0_5] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %32 = air.channel.get async [%async_token_7]  @channel_10[%c3_1, %c1_4] (%results_6[%c0_5, %c64_3] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %33 = air.channel.get async [%async_token_7]  @channel_10[%c3_1, %c2_2] (%results_6[%c0_5, %c128] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %34 = air.channel.get async [%async_token_7]  @channel_10[%c3_1, %c3_1] (%results_6[%c0_5, %c192] [%c64_3, %c64_3] [%c256_6, %c1_4]) {id = 27 : i32} : (memref<64x256xbf16, 1>)
        %async_token_8 = air.wait_all async [%async_token_4, %async_token_5, %async_token_6, %async_token_7]
        %12 = air.herd @herd_0 async [%async_token_8]  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, link_with = "mm.o", x_loc = 0 : i64, y_loc = 2 : i64} {
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
          %35 = air.channel.put async [%async_token_15]  @channel_10[%arg8, %arg9] (%results_16[%c0_14, %c0_14, %c0_14] [%c64_10, %c16, %c4_12] [%c4_12, %c256_11, %c1_13]) {id = 44 : i32} : (memref<16x16x4x4xbf16, 2>)
          %async_token_17 = air.execute [%35] {
            memref.dealloc %results_16 : memref<16x16x4x4xbf16, 2>
          }
        }
        %13 = air.channel.put async [%12]  @channel_0[] (%results_5[] [] []) {id = 45 : i32} : (memref<64x256xbf16, 1>)
        %14 = air.channel.put async [%12]  @channel_1[] (%results_6[] [] []) {id = 46 : i32} : (memref<64x256xbf16, 1>)
        %15 = air.channel.put async [%12]  @channel_2[] (%results_7[] [] []) {id = 47 : i32} : (memref<64x256xbf16, 1>)
        %16 = air.channel.put async [%12]  @channel_3[] (%results_8[] [] []) {id = 48 : i32} : (memref<64x256xbf16, 1>)
        %17 = air.wait_all async [%13, %14, %15, %16] 
        %async_token_9 = air.execute [%17] {
          memref.dealloc %results_5 : memref<64x256xbf16, 1>
        }
        %async_token_10 = air.execute [%17] {
          memref.dealloc %results_6 : memref<64x256xbf16, 1>
        }
        %async_token_11 = air.execute [%17] {
          memref.dealloc %results_7 : memref<64x256xbf16, 1>
        }
        %async_token_12 = air.execute [%17] {
          memref.dealloc %results_8 : memref<64x256xbf16, 1>
        }
      }
    }
    return
  }
}

// -----

// Wrap-and-stride list canonicalization during herd outlining.
// CHECK: aie.device(npu1)
// CHECK: %[[tile_2_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_2_1:.*]] = aie.tile(0, 1)
// CHECK: %[[tile_2_3:.*]] = aie.tile(0, 2)
// CHECK:  %[[VAL_0:.*]] = aie.mem(%[[tile_2_3]]) {
// CHECK:    %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%{{.*}}, AcquireGreaterEqual, 1)
// CHECK:    aie.dma_bd(%{{.*}} : memref<1x1x16x16x4x4xf32, 2 : i32>, 0, 4096, [<size = 64, stride = 4>, <size = 16, stride = 256>, <size = 4, stride = 1>])
// CHECK:    aie.use_lock(%{{.*}}, Release, 1)
// CHECK:    aie.next_bd ^bb1
// CHECK:  ^bb2:
// CHECK:    aie.end
// CHECK:  }
// CHECK: @func13
// RACECONDFIX: @func13

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module {
  air.channel @channel_9 [1, 1]
  air.channel @channel_8 [1, 1]
  func.func @func13(%arg0: memref<64x64xf32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<64x64xf32> attributes {id = 1 : i32} {
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg1]
        air.execute_terminator %3 : index
      }
      %async_token_1, %results_2 = air.execute -> (index) {
        %3 = affine.apply #map()[%arg2]
        air.execute_terminator %3 : index
      }
      %1 = air.channel.get async [%async_token, %async_token_1]  @channel_9[] (%arg5[%results, %results_2] [%c64, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<64x64xf32>)
      %2 = air.segment @segment0 async  {
        %c0 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        %async_token_4, %results_5 = air.execute -> (memref<1x1x64x64xf32, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xf32, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xf32, 1 : i32>
        }
        %async_token_6, %results_7 = air.execute -> (memref<1x1x16x16x4x4xf32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x16x16x4x4xf32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x16x16x4x4xf32, 2 : i32>
        }
        %3 = air.channel.get async  @channel_8[%c0, %c0] (%results_5[] [] []) {id = 16 : i32} : (memref<1x1x64x64xf32, 1 : i32>)
        %4 = air.herd @herd_0 async [%async_token_4]  tile (%arg6, %arg7) in (%arg8=%c1_3, %arg9=%c1_3) args(%arg10=%results_7) : memref<1x1x16x16x4x4xf32, 2 : i32> {
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c1_10 = arith.constant 1 : index
          %c256 = arith.constant 256 : index
          %c0_11 = arith.constant 0 : index
          %async_token_12, %results_13 = air.execute -> (index) {
            %7 = affine.apply #map1()[%arg6]
            air.execute_terminator %7 : index
          }
          %async_token_14, %results_15 = air.execute -> (index) {
            %7 = affine.apply #map1()[%arg7]
            air.execute_terminator %7 : index
          }
          %6 = air.channel.put async  @channel_8[%arg6, %arg7] (%arg10[%results_13, %c0_11, %results_15, %c0_11] [%c16, %c4, %c16, %c4] [%c16, %c4, %c256, %c1_10]) {id = 19 : i32} : (memref<1x1x16x16x4x4xf32, 2 : i32>)
        }
        %5 = air.channel.put async [%4]  @channel_9[] (%results_5[] [] []) {id = 20 : i32} : (memref<1x1x64x64xf32, 1 : i32>)
        %async_token_8 = air.execute [%4] {
          memref.dealloc %results_7 : memref<1x1x16x16x4x4xf32, 2 : i32>
        }
        %async_token_9 = air.execute [%4] {
          memref.dealloc %results_5 : memref<1x1x64x64xf32, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Unrolled bundle of channels from shim accessing directly to herd.
// CHECK: aie.device(npu1)
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK: %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK: %[[tile_1_2:.*]] = aie.tile(1, 2)
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK: %[[tile_1_3:.*]] = aie.tile(1, 3)
// CHECK: aie.flow(%[[tile_0_0]], DMA : 0, %[[tile_0_2]], DMA : 0)
// CHECK: aie.flow(%[[tile_1_0]], DMA : 0, %[[tile_1_2]], DMA : 0)
// CHECK: aie.flow(%[[tile_0_0]], DMA : 1, %[[tile_0_3]], DMA : 0)
// CHECK: aie.flow(%[[tile_1_0]], DMA : 1, %[[tile_1_3]], DMA : 0)
// CHECK: aie.shim_dma_allocation @air_channel_0_0(MM2S, 0, 0)
// CHECK: memref.global "public" @air_channel_0_0 : memref<16x8xi32, 2 : i32>
// CHECK: aie.shim_dma_allocation @air_channel_0_1(MM2S, 0, 1)
// CHECK: memref.global "public" @air_channel_0_1 : memref<16x8xi32, 2 : i32>
// CHECK: aie.shim_dma_allocation @air_channel_0_2(MM2S, 1, 0)
// CHECK: memref.global "public" @air_channel_0_2 : memref<16x8xi32, 2 : i32>
// CHECK: aie.shim_dma_allocation @air_channel_0_3(MM2S, 1, 1)
// CHECK: memref.global "public" @air_channel_0_3 : memref<16x8xi32, 2 : i32>
// CHECK: func.func @func14
// CHECK: air.channel.put  @channel_0{{.*}} metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<32x16xi32>)
// CHECK: air.channel.put  @channel_0{{.*}} metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<32x16xi32>)
// CHECK: air.channel.put  @channel_0{{.*}} metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<32x16xi32>)
// CHECK: air.channel.put  @channel_0{{.*}} metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<32x16xi32>)
// RACECONDFIX: @func14

module {
  air.channel @channel_0 [2, 2]
  func.func @func14(%arg0: memref<32x16xi32>, %arg1: memref<32x16xi32>) {
    air.launch () in () args(%arg2=%arg0) : memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      air.channel.put  @channel_0[%c0, %c0] (%arg2[%c0, %c0] [%c8, %c16] [%c32, %c1]) {id = 1 : i32} : (memref<32x16xi32>)
      air.channel.put  @channel_0[%c1, %c0] (%arg2[%c16, %c0] [%c8, %c16] [%c32, %c1]) {id = 2 : i32} : (memref<32x16xi32>)
      air.channel.put  @channel_0[%c0, %c1] (%arg2[%c0, %c16] [%c8, %c16] [%c32, %c1]) {id = 3 : i32} : (memref<32x16xi32>)
      air.channel.put  @channel_0[%c1, %c1] (%arg2[%c16, %c16] [%c8, %c16] [%c32, %c1]) {id = 4 : i32} : (memref<32x16xi32>)
      air.segment @seg  {
        %c2 = arith.constant 2 : index
        air.herd @xaddherd  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) {
          %alloc = memref.alloc() : memref<16x8xi32, 2 : i32>
          air.channel.get  @channel_0[%arg3, %arg4] (%alloc[] [] []) {id = 9 : i32} : (memref<16x8xi32, 2 : i32>)
          memref.dealloc %alloc : memref<16x8xi32, 2 : i32>
        }
      }
    }
    return
  }
}


// -----

// Ensure redundant shim DMA allocations do not occur
//
// CHECK:         aie.flow
// CHECK-NEXT: aie.shim_dma_allocation @air_channel_2(MM2S, 0, 0)
// CHECK-NEXT: memref.global "public" @air_channel_2 : memref<1024xi32, 1>
// CHECK: @func15
// RACECONDFIX: @func15
air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
func.func @func15(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  air.channel.put @channel_2[] (%arg0[] [] []) {id = 1 : i32} : (memref<1024xi32>)
  air.segment @segment0 {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %memtile0 = memref.alloc() : memref<1024xi32, 1>
    %memtile1 = memref.alloc() : memref<1024xi32, 1>
    air.channel.get @channel_2[] (%memtile0[] [] []) {id = 2 : i32} : (memref<1024xi32, 1>)
    air.channel.get @channel_2[] (%memtile1[] [] []) {id = 3 : i32} : (memref<1024xi32, 1>)
    memref.dealloc %memtile0 : memref<1024xi32, 1>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="herd4"} {
      %buf0 = memref.alloc() : memref<1024xi32, 2>
      memref.dealloc %buf0 : memref<1024xi32, 2>
    }
  }
  return
}

// -----

// Lowering complex loop structures around channel.puts/gets into BD task repeat counts (aie.mem).
//
// CHECK:      aie.mem
// CHECK-NEXT: aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK-NEXT: ^bb1:
// CHECK:      aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK:      aie.next_bd ^bb2
// CHECK-NEXT: ^bb2:
// CHECK-NEXT: aie.end
// CHECK-NEXT: ^bb3:
// CHECK-NEXT: aie.dma_start(S2MM, 0, ^bb4, ^bb2, repeat_count = 7)
// CHECK:      aie.dma_bd({{.*}}) {task_id = 1 : i32}
// CHECK:      aie.next_bd ^bb2
// CHECK:      @func16
// RACECONDFIX: @func16

air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func @func16(%arg0 : memref<5xi32>, %arg1 : memref<96xi32>, %arg2 : memref<9xi32>, %ub : index) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  air.channel.put @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<5xi32>)
  scf.for %arg3 = %c0 to %c8 step %c1 {
    air.channel.put @channel_0[] (%arg1[] [] []) {id = 2 : i32} : (memref<96xi32>)
  }
  air.segment @segment0 args (%ub_seg=%ub) : index {
    %c0_0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c8_0 = arith.constant 8 : index
    %c12_0 = arith.constant 12 : index
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %tok_5 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args (%ub_herd=%ub_seg) : index attributes { sym_name="herd4"} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c8_1 = arith.constant 8 : index
      %c12_1 = arith.constant 12 : index
      %buf0 = memref.alloc() : memref<5xi32, 2>
      %tok_6 = air.channel.get async @channel_0[] (%buf0[] [] []) {id = 7 : i32} : (memref<5xi32, 2>)
      %tok_7 = scf.for %arg5 = %c0_1 to %c8_1 step %c1_1 iter_args(%arg14 = %tok_6) -> (!air.async.token) {
        %buf1 = memref.alloc() : memref<96xi32, 2>
        %tok_5 = air.channel.get async [%arg14] @channel_0[] (%buf1[] [] []) {id = 8 : i32} : (memref<96xi32, 2>)
        memref.dealloc %buf1 : memref<96xi32, 2>
        scf.yield %tok_5 : !air.async.token
      }
      memref.dealloc %buf0 : memref<5xi32, 2>
    }
  }
  return
}

// -----

// Lowering complex loop structures around channel.puts/gets into BD task repeat counts (aie.memtile_dma).
//
// CHECK:      aie.memtile_dma
// CHECK-NEXT: aie.dma_start(MM2S, 0, ^bb1, ^bb7)
// CHECK-NEXT: ^bb1:
// CHECK:      aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK:      aie.next_bd ^bb2
// CHECK-NEXT: ^bb2:
// CHECK-NEXT: aie.end
// CHECK-NEXT: ^bb3:
// CHECK-NEXT: aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 7)
// CHECK:      aie.dma_bd({{.*}}) {task_id = 1 : i32}
// CHECK:      aie.next_bd ^bb2
// CHECK: @func17
// RACECONDFIX: @func17
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func @func17(%arg0 : memref<5xi32>, %arg1 : memref<96xi32>, %arg2 : memref<9xi32>, %ub : index) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  air.channel.put @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<5xi32>)
  scf.for %arg3 = %c0 to %c8 step %c1 {
    air.channel.put @channel_0[] (%arg1[] [] []) {id = 2 : i32} : (memref<96xi32>)
  }
  air.segment @segment0 args (%ub_seg=%ub) : index {
    %c0_0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c8_0 = arith.constant 8 : index
    %c12_0 = arith.constant 12 : index
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %memtile0 = memref.alloc() : memref<5xi32, 1>
    %tok_0 = air.channel.get async @channel_0[] (%memtile0[] [] []) {id = 3 : i32} : (memref<5xi32, 1>)
    %tok_1 = air.channel.put async [%tok_0] @channel_1[] (%memtile0[] [] []) {id = 4 : i32} : (memref<5xi32, 1>)
    %tok_2 = scf.for %arg4 = %c0_0 to %c8_0 step %c1_0 iter_args(%arg14 = %tok_1) -> (!air.async.token) {
      %memtile1 = memref.alloc() : memref<96xi32, 1>
      %tok_3 = air.channel.get async [%arg14] @channel_0[] (%memtile1[] [] []) {id = 5 : i32} : (memref<96xi32, 1>)
      %tok_4 = air.channel.put async [%tok_3] @channel_1[] (%memtile1[] [] []) {id = 6 : i32} : (memref<96xi32, 1>)
      memref.dealloc %memtile1 : memref<96xi32, 1>
      scf.yield %tok_4 : !air.async.token
    }
    memref.dealloc %memtile0 : memref<5xi32, 1>
    %tok_5 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args (%ub_herd=%ub_seg) : index attributes { sym_name="herd4"} {
      %c0_1 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c8_1 = arith.constant 8 : index
      %c12_1 = arith.constant 12 : index
      %buf0 = memref.alloc() : memref<5xi32, 2>
      %tok_6 = air.channel.get async @channel_1[] (%buf0[] [] []) {id = 7 : i32} : (memref<5xi32, 2>)
      %tok_7 = scf.for %arg5 = %c0_1 to %c8_1 step %c1_1 iter_args(%arg14 = %tok_6) -> (!air.async.token) {
        %buf1 = memref.alloc() : memref<96xi32, 2>
        %tok_5 = air.channel.get async [%arg14] @channel_1[] (%buf1[] [] []) {id = 8 : i32} : (memref<96xi32, 2>)
        memref.dealloc %buf1 : memref<96xi32, 2>
        scf.yield %tok_5 : !air.async.token
      }
      memref.dealloc %buf0 : memref<5xi32, 2>
    }
  }
  return
}

// -----

// Air.launch and air.herd only (no air.segment).
//
// CHECK:      %[[shim_noc_tile_0_0:.*]] = aie.tile(0, 0)
// CHECK:      %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK:      aie.flow(%[[tile_0_2]], DMA : 0, %[[shim_noc_tile_0_0]], DMA : 0)
// CHECK:      aie.shim_dma_allocation @air_channel_0(S2MM, 0, 0)
// CHECK:      @func18
// CHECK:      air.launch
// CHECK:      scf.for
// CHECK:      scf.for
// CHECK:      air.channel.get async{{.*}}@channel_0{{.*}}metadataArray = [{base = "air_channel_0", index = 0 : i32}]
// CHECK:      scf.yield
// CHECK:      scf.yield
// RACECONDFIX: @func18

air.channel @channel_0 [2, 2]
func.func @func18(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg8, %arg9) in (%arg10=%c1, %arg11=%c1) args(%arg12=%arg1) : memref<*xf32> attributes {id = 1 : i32} {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1_0 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = air.wait_all async 
    %2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %1) -> (!air.async.token)  : i32 {
      %async_token, %results = air.execute [%arg14] -> (i32) {
        %5 = arith.muli %arg13, %c4_i32 : i32
        air.execute_terminator %5 : i32
      }
      %4 = scf.for %arg15 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg16 = %async_token) -> (!air.async.token)  : i32 {
        %async_token_1, %results_2 = air.execute [%arg16] -> (i32) {
          %6 = arith.muli %arg15, %c2_i32 : i32
          air.execute_terminator %6 : i32
        }
        %async_token_3, %results_4 = air.execute [%arg16] -> (index) {
          %6 = arith.index_cast %results : i32 to index
          air.execute_terminator %6 : index
        }
        %async_token_5, %results_6 = air.execute [%async_token_3] -> (index) {
          %6 = arith.muli %results_4, %c128 : index
          air.execute_terminator %6 : index
        }
        %async_token_7, %results_8 = air.execute [%async_token_1] -> (index) {
          %6 = arith.index_cast %results_2 : i32 to index
          air.execute_terminator %6 : index
        }
        %async_token_9, %results_10 = air.execute [%async_token_7, %async_token_5] -> (index) {
          %6 = arith.addi %results_6, %results_8 : index
          air.execute_terminator %6 : index
        }
        %5 = air.channel.get async [%async_token_9]  @channel_0[%c0, %c0] (%arg12[%c0, %results_10] [%c4, %c2] [%c128, %c1_0]) {id = 2 : i32} : (memref<*xf32>)
        scf.yield %5 : !air.async.token
      }
      scf.yield %4 : !air.async.token
    }
    %3 = air.herd @herd_0 async  tile (%arg13, %arg14) in (%arg15=%c1_0, %arg16=%c1_0) attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
      %c1_i32_1 = arith.constant 1 : i32
      %c32_i32_2 = arith.constant 32 : i32
      %c0_i32_3 = arith.constant 0 : i32
      %4 = air.wait_all async 
      %5 = scf.for %arg17 = %c0_i32_3 to %c32_i32_2 step %c1_i32_1 iter_args(%arg18 = %4) -> (!air.async.token)  : i32 {
        %6 = scf.for %arg19 = %c0_i32_3 to %c32_i32_2 step %c1_i32_1 iter_args(%arg20 = %arg18) -> (!air.async.token)  : i32 {
          %async_token, %results = air.execute -> (memref<4x2xf32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4x2xf32, 2 : i32>
            air.execute_terminator %alloc : memref<4x2xf32, 2 : i32>
          }
          %7 = air.channel.put async [%arg20, %async_token]  @channel_0[%arg13, %arg14] (%results[] [] []) {id = 4 : i32} : (memref<4x2xf32, 2 : i32>)
          scf.yield %7 : !air.async.token
        }
        scf.yield %6 : !air.async.token
      }
    }
  }
  return
}

// -----

// Air.launch and air.herd only (no air.segment), with time-multiplexed data movement on one DMA channel.
//
// CHECK:      %[[shim_noc_tile_0_0:.*]] = aie.tile(0, 0)
// CHECK:      %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK:      %[[lock_0_2:.*]] = aie.lock(%[[tile_0_2]], 1) {init = 2
// CHECK:      %[[buf1:.*]] = aie.buffer(%[[tile_0_2]]) {sym_name = "buf1"}
// CHECK:      %[[buf0:.*]] = aie.buffer(%[[tile_0_2]]) {sym_name = "buf0"}
// CHECK:      aie.flow(%[[tile_0_2]], DMA : 0, %[[shim_noc_tile_0_0]], DMA : 0)
// CHECK:      aie.shim_dma_allocation @air_channel_0(S2MM, 0, 0)
// CHECK:      @func19
// CHECK:      air.launch
// CHECK:      scf.for
// CHECK:      scf.for
// CHECK:      air.channel.get async{{.*}}@channel_0{{.*}}metadataArray = [{base = "air_channel_0", index = 0 : i32}]
// CHECK:      scf.yield
// CHECK:      scf.yield
// RACECONDFIX: @func19

module {
  air.channel @channel_0 [2, 2]
  func.func @func19(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg8, %arg9) in (%arg10=%c1, %arg11=%c1) args(%arg12=%arg1) : memref<*xf32> attributes {id = 1 : i32} {
      %c1_i32 = arith.constant 1 : i32
      %c32_i32 = arith.constant 32 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %1) -> (!air.async.token)  : i32 {
        %async_token, %results = air.execute [%arg14] -> (i32) {
          %5 = arith.muli %arg13, %c4_i32 : i32
          air.execute_terminator %5 : i32
        }
        %4 = scf.for %arg15 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg16 = %async_token) -> (!air.async.token)  : i32 {
          %async_token_1, %results_2 = air.execute [%arg16] -> (i32) {
            %6 = arith.muli %arg15, %c2_i32 : i32
            air.execute_terminator %6 : i32
          }
          %async_token_3, %results_4 = air.execute [%arg16] -> (index) {
            %6 = arith.index_cast %results : i32 to index
            air.execute_terminator %6 : index
          }
          %async_token_5, %results_6 = air.execute [%async_token_3] -> (index) {
            %6 = arith.muli %results_4, %c128 : index
            air.execute_terminator %6 : index
          }
          %async_token_7, %results_8 = air.execute [%async_token_1] -> (index) {
            %6 = arith.index_cast %results_2 : i32 to index
            air.execute_terminator %6 : index
          }
          %async_token_9, %results_10 = air.execute [%async_token_7, %async_token_5] -> (index) {
            %6 = arith.addi %results_6, %results_8 : index
            air.execute_terminator %6 : index
          }
          %5 = air.channel.get async [%async_token_9]  @channel_0[%c0, %c0] (%arg12[%c0, %results_10] [%c4, %c2] [%c128, %c1_0]) {id = 2 : i32} : (memref<*xf32>)
          scf.yield %5 : !air.async.token
        }
        scf.yield %4 : !air.async.token
      }
      %3 = air.herd @herd_0 async  tile (%arg13, %arg14) in (%arg15=%c1_0, %arg16=%c1_0) attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_i32_1 = arith.constant 1 : i32
        %c32_i32_2 = arith.constant 32 : i32
        %c0_i32_3 = arith.constant 0 : i32
        %c2_i32_4 = arith.constant 2 : i32
        %4 = air.wait_all async 
        %5 = scf.for %arg17 = %c0_i32_3 to %c32_i32_2 step %c1_i32_1 iter_args(%arg18 = %4) -> (!air.async.token)  : i32 {
          %async_token, %results = air.execute [%arg18] -> (memref<4x2xf32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4x2xf32, 2 : i32>
            air.execute_terminator %alloc : memref<4x2xf32, 2 : i32>
          }
          %async_token_5, %results_6 = air.execute [%arg18] -> (memref<4x2xf32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4x2xf32, 2 : i32>
            air.execute_terminator %alloc : memref<4x2xf32, 2 : i32>
          }
          %6 = air.wait_all async [%async_token, %async_token_5] 
          %7 = air.wait_all async [%async_token, %async_token_5] 
          %8:4 = scf.for %arg19 = %c0_i32_3 to %c32_i32_2 step %c2_i32_4 iter_args(%arg20 = %7, %arg21 = %async_token_5, %arg22 = %6, %arg23 = %6) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token)  : i32 {
            %9 = air.channel.put async [%arg23, %arg20, %arg22]  @channel_0[%arg13, %arg14] (%results[] [] []) {id = 4 : i32} : (memref<4x2xf32, 2 : i32>)
            %10 = air.wait_all async [%arg21, %arg23, %arg20] 
            %11 = air.channel.put async [%9, %arg21]  @channel_0[%arg13, %arg14] (%results_6[] [] []) {id = 6 : i32} : (memref<4x2xf32, 2 : i32>)
            scf.yield %9, %11, %11, %10 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          scf.yield %8#1 : !air.async.token
        }
      }
    }
    return
  }
}
