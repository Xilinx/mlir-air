//===- air_channel_to_locks_ping_pong.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// one dma channel, multiple dma memcpy ops over time
// CHECK: aie.device
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {{{.*}}} : memref<32x32xbf16, 1>
// CHECK:         %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_9]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_10]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_1]])  {
// CHECK:           aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)

// CHECK:    aie.memtile_dma(%[[VAL_0]])  {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_8]] : memref<32x32xbf16, 1>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_2]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }
// CHECK: @multi_memcpys_over_time
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_0 [1, 1]
func.func @multi_memcpys_over_time() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %async_token_0, %results_1 = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %2 = air.channel.put async  @channel_0[] (%results_1[] [] []) : (memref<32x32xbf16, 1>)
      %3 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %c0 = arith.constant 0 : index
        %async_token_2, %results_3 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_4, %results_5 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %4 = air.channel.get async [%async_token_2]  @channel_0[] (%results_3[] [] []) : (memref<32x32xbf16, 2>)
        %5 = air.channel.get async [%async_token_4, %4]  @channel_0[] (%results_5[] [] []) : (memref<32x32xbf16, 2>)
        %async_token_6 = air.execute [%5] {
          memref.dealloc %results_3 : memref<32x32xbf16, 2>
        }
        %async_token_7 = air.execute [%5] {
          memref.dealloc %results_5 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// core-to-core ping pong
// CHECK: aie.device
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>

// CHECK:    aie.mem(%[[VAL_2]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_11]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_12]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_2]])  {
// CHECK:           aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_13]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_14]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_1]])  {
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK: @core_to_core_ping_pong
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_1 [1, 1]
func.func @core_to_core_ping_pong() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_4, %results_5 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_4]  @channel_1[] (%results_5[] [] []) : (memref<32x32xbf16, 2>)
          %5 = air.channel.put async [%async_token_6, %4]  @channel_1[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %5 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_4]  @channel_1[] (%results_5[] [] []) : (memref<32x32xbf16, 2>)
          %5 = air.channel.get async [%async_token_6, %4]  @channel_1[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %5 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_5 : memref<32x32xbf16, 2>
        }
        %async_token_9 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// core-to-core ping pong, with multi-token scf.for loop
// CHECK: aie.device
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>

// CHECK:    aie.mem(%[[VAL_2]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_11]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_12]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_2]])  {
// CHECK:           aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_13]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_14]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_1]])  {
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// cHECK: @core_to_core_ping_pong
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_1 [1, 1]
func.func @core_to_core_ping_pong() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %async_token_4, %results_5 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
          %4:3 = scf.for %arg13 = %c0 to %c128 step %c64 iter_args(%arg14 = %async_token_4, %arg15 = %async_token_6, %arg16 = %async_token_6) -> (!air.async.token, !air.async.token, !air.async.token) {
            %5 = air.channel.put async [%arg14, %arg16]  @channel_1[] (%results_5[] [] []) : (memref<32x32xbf16, 2>)
            %async_token_8 = air.execute [%5] {
              memref.dealloc %results_5 : memref<32x32xbf16, 2>
            }
            %6 = air.channel.put async [%5, %arg15]  @channel_1[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            %async_token_9 = air.execute [%6] {
              memref.dealloc %results_7 : memref<32x32xbf16, 2>
            }
            scf.yield %5, %6, %6 : !air.async.token, !air.async.token, !air.async.token
          }
          affine.yield %4#2 : !air.async.token
        } else {
          %4:3 = scf.for %arg13 = %c0 to %c128 step %c64 iter_args(%arg14 = %async_token_4, %arg15 = %async_token_6, %arg16 = %async_token_6) -> (!air.async.token, !air.async.token, !air.async.token) {
            %5 = air.channel.get async [%arg14, %arg16]  @channel_1[] (%results_5[] [] []) : (memref<32x32xbf16, 2>)
            %async_token_8 = air.execute [%5] {
              memref.dealloc %results_5 : memref<32x32xbf16, 2>
            }
            %6 = air.channel.get async [%5, %arg15]  @channel_1[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            %async_token_9 = air.execute [%6] {
              memref.dealloc %results_7 : memref<32x32xbf16, 2>
            }
            scf.yield %5, %6, %6 : !air.async.token, !air.async.token, !air.async.token
          }
          affine.yield %4#2 : !air.async.token
        }
      }
    }
  }
  return
}

// -----

// ping-pong is not possible with multiple channel accesses to the same buffer, due to dependence arising from the prod. and cons. of data in the buffer.
// CHECK: aie.device
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = aie.tile(0, 3)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_0]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 1) {init = 1 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_11:.*]] = aie.buffer(%[[VAL_0]]) {{{.*}}} : memref<1x1x64x32xi32, 1 : i32>
// CHECK:         %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<1x1x4x8x4x8xi32, 2 : i32>

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_12]] : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024) {task_id = 0 : i32}
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:  // 3 preds: ^bb1, ^bb3, ^bb4
// CHECK:           aie.end
// CHECK:         ^bb3:
// CHECK:           aie.dma_start(S2MM, 0, ^bb4, ^bb2, repeat_count = 3)
// CHECK:         ^bb4:  // pred: ^bb3
// CHECK:           aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_12]] : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024) {task_id = 1 : i32}
// CHECK:           aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         }

// CHECK:    aie.core(%[[VAL_1]])  {
// CHECK:         cf.br ^bb1
// CHECK:       ^bb1:  // pred: ^bb0
// CHECK:         cf.br ^bb2
// CHECK:       ^bb2:  // pred: ^bb1
// CHECK:         aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:         cf.br ^bb3
// CHECK:       ^bb3:  // pred: ^bb2
// CHECK:         cf.br ^bb4
// CHECK:       ^bb4:  // pred: ^bb3
// CHECK:         scf.for %arg0 = %c1 to %c5 step %c1 {
// CHECK:           aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:         }
// CHECK:         aie.end

// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// cHECK: @not_really_ping_pong

air.channel @channel_2 [1, 1]
func.func @not_really_ping_pong() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
    %6 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 3 : i64, y_size = 4 : i64} {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c5_1 = arith.constant 5 : index
      %c32_21 = arith.constant 32 : index
      %c64_22 = arith.constant 64 : index
      %c0_23 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_24 = arith.constant 1 : index
      %async_token_31, %results_32 = air.execute -> (memref<1x1x4x8x4x8xi32, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
      }
      %7 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c1_24, %arg10=%c1_24) args(%arg11=%results_32) : memref<1x1x4x8x4x8xi32, 2 : i32> attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
        %23 = air.channel.get async  @channel_2[] (%arg11[] [] []) {id = 4 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
      }
      %async_token_39, %results_40 = air.execute -> (memref<1x1x64x32xi32, 1 : i32>) {
        %alloc = memref.alloc() : memref<1x1x64x32xi32, 1 : i32>
        air.execute_terminator %alloc : memref<1x1x64x32xi32, 1 : i32>
      }
      %9 = scf.for %arg7 = %c0_23 to %c5_1 step %c1_24 iter_args(%arg8 = %async_token_39) -> (!air.async.token) {
        %26 = air.channel.put async [%arg8] @channel_2[] (%results_40[%c0_23, %c0_23, %c0_23] [%c4, %c32_21, %c8] [%c8, %c32_21, %c1_24]) {id = 13 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        scf.yield %26 : !air.async.token
      }
      %async_token_52 = air.execute  {
        memref.dealloc %results_40 : memref<1x1x64x32xi32, 1 : i32>
      }
      %17 = air.herd @herd_0 async [%7]  tile (%arg7, %arg8) in (%arg9=%c1_24, %arg10=%c1_24) args(%arg11=%results_32) : memref<1x1x4x8x4x8xi32, 2 : i32> attributes {id = 4 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
        %c5_51 = arith.constant 5 : index
        %c1_52 = arith.constant 1 : index
        scf.for %arg14 = %c1_52 to %c5_51 step %c1_52 {
          %23 = air.channel.get async  @channel_2[] (%arg11[] [] []) {id = 26 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
        }
      }
      %async_token_47 = air.execute {
        memref.dealloc %results_32 : memref<1x1x4x8x4x8xi32, 2 : i32>
      }
    }
  }
  return
}
