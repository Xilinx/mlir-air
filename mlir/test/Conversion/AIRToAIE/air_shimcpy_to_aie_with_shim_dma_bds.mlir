//===- air_shimcpy_to_aie_with_shim_dma_bds.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=2 device=xcvc1902 generate-shim-dma=true" --split-input-file | FileCheck %s

// air.dma_memcpy_nd to aie.locks.
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_1]], 0)
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<1024xi32, 2>

// CHECK:    AIE.mem(%[[VAL_1]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_5]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_1]])  {

// CHECK:         AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)

// CHECK:    AIE.shimDMA(%[[VAL_2]])  {
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_0]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }
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

// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_1:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 0 : i32}
// CHECK:         %[[VAL_5:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_6:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_8:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_9:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_2]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_8]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_9]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_2]])  {
// CHECK:           AIE.useLock(%[[VAL_7]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_6]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_7]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_6]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_3]], DMA : 1, %[[VAL_2]], DMA : 1)

// CHECK:    AIE.shimDMA(%[[VAL_3]])  {
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_1]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_0]] : memref<1024xi32>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 0)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

func.func @func2(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
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
    air.dma_memcpy_nd (%buf1[] [] [], %ext0[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<512xi32, 2>, memref<1024xi32>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    memref.dealloc %buf1 : memref<512xi32, 2>
    air.herd_terminator
  }
  return
}

// -----

// air.channel to aie.locks.
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_1:.*]] = AIE.external_buffer {sym_name = {{.*}}} : memref<1024xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_2]], 1)
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:         %[[VAL_5:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_6:.*]] = AIE.lock(%[[VAL_5]], 1)
// CHECK:         %[[VAL_7:.*]] = AIE.lock(%[[VAL_5]], 0)
// CHECK:         %[[VAL_8:.*]] = AIE.buffer(%[[VAL_5]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_9:.*]] = AIE.buffer(%[[VAL_5]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_5]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_8]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_7]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_9]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_7]], Release, 0)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_5]])  {
// CHECK:           AIE.useLock(%[[VAL_6]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_7]], Acquire, 0)
// CHECK:           AIE.useLock(%[[VAL_6]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_5]], DMA : 0, %[[VAL_2]], DMA : 0)

// CHECK:    AIE.shimDMA(%[[VAL_2]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_1]] : memref<1024xi32>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_0]] : memref<1024xi32>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

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
