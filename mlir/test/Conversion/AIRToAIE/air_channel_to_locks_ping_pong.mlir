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
// CHECK:         %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = {{.*}}} : memref<32x32xbf16, 1>
// CHECK:         %[[VAL_9:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_10:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>

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
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// core-to-core ping pong
// CHECK: aie.device
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>

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
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// core-to-core ping pong, with multi-token scf.for loop
// CHECK: aie.device
// CHECK:         %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_7:.*]] = aie.lock(%[[VAL_2]], 1) {init = 2 : i32}
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_2]], 0) {init = 0 : i32}
// CHECK:         %[[VAL_11:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>

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
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
