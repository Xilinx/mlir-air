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
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%buf0[] [] [], %ext0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>)
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
// CHECK:           AIE.useLock(%[[VAL_14]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_15]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_14]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_15]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_10]], DMA : 0, %[[VAL_12]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_10]], DMA : 1, %[[VAL_12]], DMA : 1)
func.func @func2(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func2"} {
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
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_1]], 1)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_1]])  {
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

// CHECK:    AIE.core(%[[VAL_1]])  {
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
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
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func3"} {
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
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_1]], 1)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_1]])  {
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

// CHECK:    AIE.core(%[[VAL_1]])  {
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)

air.channel @channel_2 [1, 1]
air.channel @channel_3 [1, 1]
func.func @func4(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.channel.put @channel_2[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
  air.channel.put @channel_3[] (%arg1[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32>)
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func4"} {
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
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_1]], 1)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 0)
// CHECK:         %[[VAL_4:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<512xi32, 2>

// CHECK:    AIE.mem(%[[VAL_1]])  {
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

// CHECK:    AIE.core(%[[VAL_1]])  {
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)

air.channel @channel_4 [1, 1]
air.channel @channel_5 [1, 1]
func.func @func5(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  %token_0 = air.channel.put async @channel_4[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
  %token_1 = air.channel.put async [%token_0] @channel_5[] (%arg1[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32>)
  %token_2 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func5"} {
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

// -----

// L3 to L1 broadcast
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(3, 2)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(4, 2)
// CHECK:         %[[VAL_4:.*]] = AIE.tile(5, 2)
// CHECK:         %[[VAL_5:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_6:.*]] = AIE.tile(3, 3)
// CHECK:         %[[VAL_7:.*]] = AIE.tile(4, 3)
// CHECK:         %[[VAL_8:.*]] = AIE.tile(5, 3)
// CHECK:         %[[VAL_9:.*]] = AIE.tile(2, 4)
// CHECK:         %[[VAL_10:.*]] = AIE.tile(3, 4)
// CHECK:         %[[VAL_11:.*]] = AIE.tile(4, 4)
// CHECK:         %[[VAL_12:.*]] = AIE.tile(5, 4)
// CHECK:         %[[VAL_13:.*]] = AIE.tile(2, 5)
// CHECK:         %[[VAL_14:.*]] = AIE.tile(3, 5)
// CHECK:         %[[VAL_15:.*]] = AIE.tile(4, 5)
// CHECK:         %[[VAL_16:.*]] = AIE.tile(5, 5)

// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_9]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_13]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 1)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_2]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_4]], DMA : 0)

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
air.channel @channel_6 [1, 1] {broadcast_shape = [1, 4]}
air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
func.func @func6(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %c512 = arith.constant 0 : index
  %c1024 = arith.constant 0 : index
  %herd_cols = arith.constant 4 : index
  %herd_rows = arith.constant 4 : index
  %token_0 = air.channel.put async @channel_6[] (%arg0[%c0] [%c1024] [%c1]) {id = 1 : i32} : (memref<1024xi32>)
  %token_1 = air.channel.put async [%token_0] @channel_7[] (%arg1[%c0] [%c512] [%c1]) {id = 2 : i32} : (memref<1024xi32>)
  %token_2 = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="func6"} {
    %token_3, %buf0 = air.execute -> (memref<1024xi32, 2>) {
      %alloc = memref.alloc() : memref<1024xi32, 2>
      air.execute_terminator %alloc : memref<1024xi32, 2>
    }
    %token_4, %buf1 = air.execute -> (memref<512xi32, 2>) {
      %alloc = memref.alloc() : memref<512xi32, 2>
      air.execute_terminator %alloc : memref<512xi32, 2>
    }
    %aif0 = affine.if #set()[%tx, %ty] -> !air.async.token {
      %17 = air.channel.get async [%token_3, %token_4]  @channel_6[%tx, %ty] (%buf0[] [] []) {id = 3 : i32} : (memref<1024xi32, 2>)
      affine.yield %17 : !air.async.token
    } else {
      %17 = air.wait_all async [%token_3, %token_4]
      affine.yield %17 : !air.async.token
    }
    %aif1 = affine.if #set1()[%tx, %ty] -> !air.async.token {
      %17 = air.channel.get async [%aif0]  @channel_7[%tx, %ty] (%buf1[] [] []) {id = 4 : i32} : (memref<512xi32, 2>)
      affine.yield %17 : !air.async.token
    } else {
      %17 = air.wait_all async [%aif0]
      affine.yield %17 : !air.async.token
    }
    %token_7 = air.execute [%aif1] {
      memref.dealloc %buf0 : memref<1024xi32, 2>
    }
    %token_8 = air.execute [%aif1] {
      memref.dealloc %buf1 : memref<512xi32, 2>
    }
    air.herd_terminator
  }
  return
}

// -----

// DMA bd program taking into account hoisted partial pixel copies
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32}
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32}
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32}
// CHECK:         %[[VAL_5:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_6:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_7:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<1024xi32, 2>
// CHECK:         %[[VAL_8:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<1024xi32, 2>

// CHECK:    AIE.mem(%[[VAL_0]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb6)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         ^bb3:
// CHECK:           AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
// CHECK:         ^bb4:
// CHECK:           AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_7]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:           AIE.nextBd ^bb5
// CHECK:         ^bb5:
// CHECK:           AIE.useLock(%[[VAL_2]], Acquire, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_8]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_2]], Release, 1)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         ^bb6:
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb7, ^bb3)
// CHECK:         ^bb7:
// CHECK:           AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_6]] : memref<1024xi32, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:           AIE.nextBd ^bb7
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_0]])  {
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 1)
// CHECK:           scf.for
// CHECK:             AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_2]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:           }
// CHECK:           AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:           AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_0]], DMA : 1)
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)

air.channel @channel_8 [1, 1]
air.channel @channel_9 [1, 1]
air.channel @channel_10 [1, 1]
air.channel @channel_11 [1, 1]
func.func @func7(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>, %arg2 : memref<1024xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c96 = arith.constant 96 : index
  %c384 = arith.constant 384 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  %token_0 = air.channel.put async @channel_8[] (%arg0[] [] []) {id = 1 : i32} : (memref<1024xi32>)
  %for_loop = scf.for %arg7 = %c0 to %c384 step %c96 iter_args(%arg8 = %token_0) -> (!air.async.token) {
    %token_1 = air.channel.put async [%arg8] @channel_9[] (%arg1[] [] []) {id = 2 : i32} : (memref<1024xi32>)
    %token_2 = air.channel.put async [%token_1] @channel_10[] (%arg2[] [] []) {id = 3 : i32} : (memref<1024xi32>)
    scf.yield %token_2 : !air.async.token
  }
  %token_11 = air.channel.get async [%for_loop] @channel_11[] (%arg0[] [] []) {id = 1 : i32} : (memref<1024xi32>)
  %herd_token = air.herd async tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="func7"} {
    %c0_1 = arith.constant 0 : index
    %c96_3 = arith.constant 96 : index
    %c384_2 = arith.constant 384 : index
    %token_3, %buf0 = air.execute -> (memref<1024xi32, 2>) {
      %alloc = memref.alloc() : memref<1024xi32, 2>
      air.execute_terminator %alloc : memref<1024xi32, 2>
    }
    %token_4 = air.channel.get async [%token_3] @channel_8[%tx, %ty] (%buf0[] [] []) {id = 4 : i32} : (memref<1024xi32, 2>)
    %for_loop_1 = scf.for %arg7 = %c0_1 to %c384_2 step %c96_3 iter_args(%arg8 = %token_4) -> (!air.async.token) {
      %token_5, %buf1 = air.execute [%arg8] -> (memref<1024xi32, 2>) {
        %alloc = memref.alloc() : memref<1024xi32, 2>
        air.execute_terminator %alloc : memref<1024xi32, 2>
      }
      %token_6, %buf2 = air.execute [%arg8] -> (memref<1024xi32, 2>) {
        %alloc = memref.alloc() : memref<1024xi32, 2>
        air.execute_terminator %alloc : memref<1024xi32, 2>
      }
      %token_7 = air.channel.get async [%token_5, %token_6] @channel_9[%tx, %ty] (%buf1[] [] []) {id = 5 : i32} : (memref<1024xi32, 2>)
      %token_8 = air.channel.get async [%token_7] @channel_10[%tx, %ty] (%buf2[] [] []) {id = 6 : i32} : (memref<1024xi32, 2>)
      %token_9 = air.execute [%token_8] {
        memref.dealloc %buf2 : memref<1024xi32, 2>
      }
      %token_10 = air.execute [%token_8] {
        memref.dealloc %buf1 : memref<1024xi32, 2>
      }
      %wait = air.wait_all async [%token_9, %token_10]
      scf.yield %wait : !air.async.token
    }
    %token_12 = air.channel.put async [%for_loop_1] @channel_11[] (%buf0[] [] []) {id = 1 : i32} : (memref<1024xi32, 2>)
    %token_13 = air.execute [%token_12] {
      memref.dealloc %buf0 : memref<1024xi32, 2>
    }
    air.herd_terminator
  }
  return
}
