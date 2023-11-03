//===- air_channel_to_locks_core_to_core.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// one-to-one communication
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(2, 4)
// CHECK:         %[[VAL_3:.*]] = AIE.lock(%[[VAL_1]], 1)
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_1]], 0)
// CHECK:         %[[VAL_5:.*]] = AIE.lock(%[[VAL_2]], 1)
// CHECK:         %[[VAL_6:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:         %[[VAL_7:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_8:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>

// CHECK:    AIE.mem(%[[VAL_2]])  {
// CHECK:           AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_7]] : memref<32x32xbf16, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_2]])  {
// CHECK:           AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.mem(%[[VAL_1]])  {
// CHECK:           AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_8]] : memref<32x32xbf16, 2>, 0, 1024>, 0)
// CHECK:           AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:           AIE.nextBd ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_1]])  {
// CHECK:           AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_0 [1, 1]
func.func @one_to_one() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
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

// two-to-two parallel dataflow
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(3, 3)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(2, 4)
// CHECK:         %[[VAL_4:.*]] = AIE.tile(3, 4)
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_15:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_16:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>

// CHECK:         AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)

#set1 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
air.channel @channel_1 [1, 2]
func.func @two_to_two() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_1[%c0, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_1[%c0, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
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

// one-to-two core-to-core broadcast
// CHECK: AIE.device
// CHECK:         %[[VAL_0:.*]] = AIE.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(3, 3)
// CHECK:         %[[VAL_3:.*]] = AIE.tile(2, 4)
// CHECK:         %[[VAL_4:.*]] = AIE.tile(3, 4)
// CHECK:         %[[VAL_13:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_15:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_16:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = {{.*}}} : memref<32x32xbf16, 2>

// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:         AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 1)

#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 == 0, s1 == 1)>
air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
air.channel @channel_3 [1, 1] {broadcast_shape = [1, 2]}
func.func @one_to_two() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_2[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %5 = affine.if #set3()[%arg8, %arg9] -> !air.async.token {
            %6 = air.channel.put async [%async_token_6]  @channel_3[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %6 : !air.async.token
          } else {
            %6 = air.channel.get async [%async_token_6]  @channel_2[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            %7 = air.channel.get async [%6]  @channel_3[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %7 : !air.async.token
          }
          affine.yield %5 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
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
