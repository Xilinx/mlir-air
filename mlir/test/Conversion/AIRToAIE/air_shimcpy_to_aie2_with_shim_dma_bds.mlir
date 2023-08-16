//===- air_shimcpy_to_aie2_with_shim_dma_bds.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802 generate-shim-dma=true" --split-input-file | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:  %[[VAL_0:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:  %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:  %[[VAL_2:.*]] = AIE.tile(2, 0)
// CHECK:  %[[VAL_3:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_4:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_5:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:  %[[VAL_6:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_7:.*]] = AIE.buffer(%[[VAL_1]]) {{.*}} : memref<1024xi32, 2>
// CHECK:  AIE.mem(%[[VAL_1]]) {
// CHECK:    AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:    AIE.dmaBd(<%[[VAL_7]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:    AIE.nextBd ^bb1
// CHECK:  ^bb2:
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  AIE.core(%[[VAL_1]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:    AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:    AIE.end
// CHECK:  AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:  AIE.shimDMA(%[[VAL_2]]) {
// CHECK:    AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:  ^bb1:
// CHECK:    AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:    AIE.dmaBd(<%[[VAL_0]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:    AIE.nextBd ^bb1
// CHECK:  ^bb2:
// CHECK:    AIE.end
// CHECK:  }
func.func @func1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%buf0[] [] [], %ext0[%c0] [%c1024] [%c1]) : (memref<1024xi32, 2>, memref<1024xi32>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    air.herd_terminator
  }
  return
}

// -----

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK: %[[VAL_0:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK: %[[VAL_1:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<512xi32>
// CHECK: %[[VAL_2:.*]] = AIE.tile(2, 3)
// CHECK: %[[VAL_3:.*]] = AIE.tile(2, 0)
// CHECK: %[[VAL_4:.*]] = AIE.lock(%[[VAL_3]], 3) {init = 1 : i32}
// CHECK: %[[VAL_5:.*]] = AIE.lock(%[[VAL_3]], 2) {init = 0 : i32}
// CHECK: %[[VAL_6:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 1 : i32}
// CHECK: %[[VAL_7:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK: %[[VAL_8:.*]] = AIE.lock(%[[VAL_2]], 3) {init = 1 : i32}
// CHECK: %[[VAL_9:.*]] = AIE.lock(%[[VAL_2]], 2) {init = 0 : i32}
// CHECK: %[[VAL_10:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 1 : i32}
// CHECK: %[[VAL_11:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK: %[[VAL_12:.*]] = AIE.buffer(%[[VAL_2]]) {{.*}} : memref<1024xi32, 2>
// CHECK: %[[VAL_13:.*]] = AIE.buffer(%[[VAL_2]]) {{.*}} : memref<512xi32, 2>
// CHECK: AIE.mem(%[[VAL_2]]) {
// CHECK:   AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_12]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb2:
// CHECK:   AIE.end
// CHECK: ^bb3:
// CHECK:   AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_13]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:   AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: }
// CHECK: AIE.core(%[[VAL_2]]) {
// CHECK:   cf.br ^bb1
// CHECK: ^bb1:
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   AIE.useLock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:   AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:   AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:   AIE.end
// CHECK: AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK: AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK: AIE.shimDMA(%[[VAL_3]]) {
// CHECK:   AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_1]] : memref<512xi32>, 0, 512>, 0)
// CHECK:   AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb2:
// CHECK:   AIE.end
// CHECK: ^bb3:
// CHECK:   AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_0]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: }
func.func @func2(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : index
    %c1024 = arith.constant 0 : index
    %c512 = arith.constant 0 : index
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
// CHECK:         %[[VAL_0:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_1:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<512xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_3]], 3)
// CHECK:         %[[VAL_5:.*]] = AIE.lock(%[[VAL_3]], 2)
// CHECK:         %[[VAL_6:.*]] = AIE.lock(%[[VAL_3]], 1)
// CHECK:         %[[VAL_7:.*]] = AIE.lock(%[[VAL_3]], 0)
// CHECK:         %[[VAL_8:.*]] = AIE.lock(%[[VAL_2]], 3)
// CHECK:         %[[VAL_9:.*]] = AIE.lock(%[[VAL_2]], 2)
// CHECK:         %[[VAL_10:.*]] = AIE.lock(%[[VAL_2]], 1)
// CHECK:         %[[VAL_11:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:         %[[VAL_12:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_2]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_12]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_13]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_2]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:           AIE.useLock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:           AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK: AIE.shimDMA(%[[VAL_3]]) {
// CHECK:   AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_1]] : memref<512xi32>, 0, 512>, 0)
// CHECK:   AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:   AIE.nextBd ^bb1
// CHECK: ^bb2:
// CHECK:   AIE.end
// CHECK: ^bb3:
// CHECK:   AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK: ^bb4:
// CHECK:   AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:   AIE.dmaBd(<%[[VAL_0]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:   AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:   AIE.nextBd ^bb4
// CHECK: }

air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func @func3(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %c512 = arith.constant 0 : index
  %c1024 = arith.constant 0 : index
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
