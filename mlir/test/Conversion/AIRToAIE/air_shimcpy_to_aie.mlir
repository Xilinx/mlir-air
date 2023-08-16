//===- air_shimcpy_to_aie.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=2 device=xcvc1902" --split-input-file | FileCheck %s

// air.dma_memcpy_nd to aie.locks.
// CHECK: AIE.device
// CHECK:         %[[VAL_12:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_10:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_14:.*]] = AIE.lock(%[[VAL_12]], 0)
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_12]]) {sym_name = {{.*}}} : memref<1024xi32, 2>

// CHECK:    AIE.mem(%[[VAL_12]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_13]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_12]])  {
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_10]], DMA : 0, %[[VAL_12]], DMA : 0)
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

// CHECK: AIE.device
// CHECK:         %[[VAL_12:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_10:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_15:.*]] = AIE.lock(%[[VAL_12]], 1)
// CHECK:         %[[VAL_14:.*]] = AIE.lock(%[[VAL_12]], 0)
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_12]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_16:.*]] = AIE.buffer(%[[VAL_12]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_12]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_13]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_15]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_16]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_12]])  {
// CHECK:           AIE.useLock(%[[VAL_15]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_15]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_10]], DMA : 0, %[[VAL_12]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_10]], DMA : 1, %[[VAL_12]], DMA : 1)
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
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 1)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_0]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_4]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_5]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_0]])  {
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)

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

// -----

// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 1)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_0]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_4]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_5]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_0]])  {
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], DMA : 1)

air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
func.func @func4(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %c512 = arith.constant 0 : index
  %c1024 = arith.constant 0 : index
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.channel.put @channel_2[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
  air.channel.put @channel_3[] (%arg1[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32>)
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    %buf1 = memref.alloc() : memref<512xi32, 2>
    air.channel.get @channel_2[%tx, %ty] (%buf0[] [] []) {id = 3 : i32} : (memref<1024xi32, 2>)
    air.channel.get @channel_3[%tx, %ty] (%buf1[] [] []) {id = 4 : i32} : (memref<512xi32, 2>)
    memref.dealloc %buf0 : memref<1024xi32, 2>
    memref.dealloc %buf1 : memref<512xi32, 2>
    air.herd_terminator
  }
  return
}

// -----

// asynchronous air.channel to aie.locks.
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 1)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_0]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_4]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_5]] : memref<512xi32, 2>, 0, 512>, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_0]])  {
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], DMA : 1)

air.channel @channel_4 [1, 1]
air.channel @channel_5 [1, 1]
func.func @func5(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %c512 = arith.constant 0 : index
  %c1024 = arith.constant 0 : index
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  %token_0 = air.channel.put async @channel_4[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
  %token_1 = air.channel.put async [%token_0] @channel_5[] (%arg1[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32>)
  %token_2 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
    %token_3, %buf0 = air.execute -> (memref<1024xi32, 2>) {
      %alloc = memref.alloc() : memref<1024xi32, 2>
      air.execute_terminator %alloc : memref<1024xi32, 2>
    }
    %token_4, %buf1 = air.execute -> (memref<512xi32, 2>) {
      %alloc = memref.alloc() : memref<512xi32, 2>
      air.execute_terminator %alloc : memref<512xi32, 2>
    }
    %token_5 = air.channel.get async [%token_3, %token_4] @channel_4[%tx, %ty] (%buf0[] [] []) {id = 3 : i32} : (memref<1024xi32, 2>)
    %token_6 = air.channel.get async [%token_5] @channel_5[%tx, %ty] (%buf1[] [] []) {id = 4 : i32} : (memref<512xi32, 2>)
    %token_7 = air.execute [%token_6] {
      memref.dealloc %buf0 : memref<1024xi32, 2>
    }
    %token_8 = air.execute [%token_6] {
      memref.dealloc %buf1 : memref<512xi32, 2>
    }
    air.herd_terminator
  }
  return
}

// TODO: channel broadcast

// air.channel @channel_6 [1, 1] {broadcast_shape = [1, 4]}
// air.channel @channel_7 [1, 1] {broadcast_shape = [1, 4]}
// func.func @func6(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 0 : index
//   %c512 = arith.constant 0 : index
//   %c1024 = arith.constant 0 : index
//   %herd_cols = arith.constant 1 : index
//   %herd_rows = arith.constant 4 : index
//   %token_0 = air.channel.put async @channel_6[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
//   %token_1 = air.channel.put async [%token_0] @channel_7[] (%arg1[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32>)
//   %token_2 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func1"} {
//     %token_3, %buf0 = air.execute -> (memref<1024xi32, 2>) {
//       %alloc = memref.alloc() : memref<1024xi32, 2>
//       air.execute_terminator %alloc : memref<1024xi32, 2>
//     }
//     %token_4, %buf1 = air.execute -> (memref<512xi32, 2>) {
//       %alloc = memref.alloc() : memref<512xi32, 2>
//       air.execute_terminator %alloc : memref<512xi32, 2>
//     }
//     %token_5 = air.channel.get async [%token_3, %token_4] @channel_6[%tx, %ty] (%buf0[] [] []) {id = 3 : i32} : (memref<1024xi32, 2>)
//     %token_6 = air.channel.get async [%token_5] @channel_7[%tx, %ty] (%buf1[] [] []) {id = 4 : i32} : (memref<512xi32, 2>)
//     %token_7 = air.execute [%token_6] {
//       memref.dealloc %buf0 : memref<1024xi32, 2>
//     }
//     %token_8 = air.execute [%token_6] {
//       memref.dealloc %buf1 : memref<512xi32, 2>
//     }
//     air.herd_terminator
//   }
//   return
// }
