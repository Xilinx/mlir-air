//===- partition_memref_empty_offsets.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for GitHub issue #1389: partitionMemref must not crash when
// channel ops have empty offsets (default full-memref access pattern). When an
// L2 memref has more unique channel connections than the memtile hardware limit,
// specializeL2MemrefsIntoMemtiles calls partitionMemref. If any channel op has
// empty offsets, partitionMemref should return early instead of crashing on
// getOffsets().front().

// RUN: air-opt %s -air-to-aie='device=npu1' | FileCheck %s

// The L2 buffer should remain as a single unpartitioned buffer on the memtile,
// because the empty-offset channel.put prevents partitioning.
// CHECK-LABEL: aie.device(npu1)
// CHECK:         %[[MEMTILE:.*]] = aie.tile(1, 1)
// CHECK:         aie.buffer(%[[MEMTILE]]) {{{.*}}} : memref<256x256xbf16, 1>
// CHECK-NOT:     aie.buffer(%[[MEMTILE]]) {{{.*}}} : memref<{{.*}}xbf16, 1>

module {
  // Seven S2MM channels on the L2 buffer (exceeds memtile DMA limit of 6).
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_5 [1, 1]
  air.channel @channel_6 [1, 1]
  // One MM2S channel with empty offsets (the trigger for the crash).
  air.channel @channel_out [1, 1]
  // L1 channels.
  air.channel @channel_l1_0 [1, 1]
  air.channel @channel_l1_1 [1, 1]
  air.channel @channel_l1_2 [1, 1]
  air.channel @channel_l1_3 [1, 1]
  air.channel @channel_l1_4 [1, 1]
  air.channel @channel_l1_5 [1, 1]
  air.channel @channel_l1_6 [1, 1]
  func.func @partition_memref_empty_offsets(%arg0: memref<256x256xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%la0, %la1, %la2) in (%la3=%c1, %la4=%c1, %la5=%c1) args(%ext=%arg0) : memref<256x256xbf16> attributes {id = 1 : i32} {
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      // L3 -> L2: channel_out will carry the result back (empty offsets).
      %1 = air.channel.get async  @channel_out[] (%ext[] [] []) {id = 1 : i32} : (memref<256x256xbf16>)
      %2 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c4 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c192 = arith.constant 192 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        %c256_2 = arith.constant 256 : index
        %c0_3 = arith.constant 0 : index
        %c64_4 = arith.constant 64 : index
        // L2 buffer with >6 unique channel connections.
        %async_token, %results = air.execute -> (memref<256x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<256x256xbf16, 1>
          air.execute_terminator %alloc : memref<256x256xbf16, 1>
        }
        // 7 channel.get ops with explicit offsets (S2MM, writing tiles into L2).
        %3 = air.channel.get async [%async_token]  @channel_0[] (%results[%c0_3, %c0_3] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 2 : i32} : (memref<256x256xbf16, 1>)
        %4 = air.channel.get async [%async_token]  @channel_1[] (%results[%c0_3, %c64_4] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 3 : i32} : (memref<256x256xbf16, 1>)
        %5 = air.channel.get async [%async_token]  @channel_2[] (%results[%c0_3, %c128] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 4 : i32} : (memref<256x256xbf16, 1>)
        %6 = air.channel.get async [%async_token]  @channel_3[] (%results[%c0_3, %c192] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 5 : i32} : (memref<256x256xbf16, 1>)
        %7 = air.channel.get async [%async_token]  @channel_4[] (%results[%c64_4, %c0_3] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 6 : i32} : (memref<256x256xbf16, 1>)
        %8 = air.channel.get async [%async_token]  @channel_5[] (%results[%c64_4, %c64_4] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 7 : i32} : (memref<256x256xbf16, 1>)
        %9 = air.channel.get async [%async_token]  @channel_6[] (%results[%c64_4, %c128] [%c64_4, %c64_4] [%c256_2, %c1_1]) {id = 8 : i32} : (memref<256x256xbf16, 1>)
        // 1 channel.put with EMPTY offsets (MM2S, sending full L2 buffer out).
        // This is the op that triggers the crash in partitionMemref.
        %10 = air.channel.put async [%3, %4, %5, %6, %7, %8, %9]  @channel_out[] (%results[] [] []) {id = 9 : i32} : (memref<256x256xbf16, 1>)
        %async_token_5 = air.execute [%10] {
          memref.dealloc %results : memref<256x256xbf16, 1>
        }
        // Herd producing data via L1 channels.
        %11 = air.herd @herd_0 async  tile (%tx, %ty) in (%sx=%c4, %sy=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0_6 = arith.constant 0 : index
          %async_token_7, %results_8 = air.execute -> (memref<64x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2>
            air.execute_terminator %alloc : memref<64x64xbf16, 2>
          }
          %12 = affine.if affine_set<()[s0, s1] : (s0 == 0, s1 == 0)>()[%tx, %ty] -> !air.async.token {
            %13 = air.channel.put async [%async_token_7]  @channel_0[%tx, %ty] (%results_8[] [] []) {id = 10 : i32} : (memref<64x64xbf16, 2>)
            affine.yield %13 : !air.async.token
          } else {
            %13 = affine.if affine_set<()[s0, s1] : (s0 == 0, s1 - 1 == 0)>()[%tx, %ty] -> !air.async.token {
              %14 = air.channel.put async [%async_token_7]  @channel_1[%tx, %ty] (%results_8[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2>)
              affine.yield %14 : !air.async.token
            } else {
              %14 = affine.if affine_set<()[s0, s1] : (s0 == 0, s1 - 2 == 0)>()[%tx, %ty] -> !air.async.token {
                %15 = air.channel.put async [%async_token_7]  @channel_2[%tx, %ty] (%results_8[] [] []) {id = 12 : i32} : (memref<64x64xbf16, 2>)
                affine.yield %15 : !air.async.token
              } else {
                %15 = affine.if affine_set<()[s0, s1] : (s0 == 0, s1 - 3 == 0)>()[%tx, %ty] -> !air.async.token {
                  %16 = air.channel.put async [%async_token_7]  @channel_3[%tx, %ty] (%results_8[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 2>)
                  affine.yield %16 : !air.async.token
                } else {
                  %16 = affine.if affine_set<()[s0, s1] : (s0 - 1 == 0, s1 == 0)>()[%tx, %ty] -> !air.async.token {
                    %17 = air.channel.put async [%async_token_7]  @channel_4[%tx, %ty] (%results_8[] [] []) {id = 14 : i32} : (memref<64x64xbf16, 2>)
                    affine.yield %17 : !air.async.token
                  } else {
                    %17 = affine.if affine_set<()[s0, s1] : (s0 - 1 == 0, s1 - 1 == 0)>()[%tx, %ty] -> !air.async.token {
                      %18 = air.channel.put async [%async_token_7]  @channel_5[%tx, %ty] (%results_8[] [] []) {id = 15 : i32} : (memref<64x64xbf16, 2>)
                      affine.yield %18 : !air.async.token
                    } else {
                      %18 = air.channel.put async [%async_token_7]  @channel_6[%tx, %ty] (%results_8[] [] []) {id = 16 : i32} : (memref<64x64xbf16, 2>)
                      affine.yield %18 : !air.async.token
                    }
                    affine.yield %17 : !air.async.token
                  }
                  affine.yield %16 : !air.async.token
                }
                affine.yield %15 : !air.async.token
              }
              affine.yield %14 : !air.async.token
            }
            affine.yield %13 : !air.async.token
          }
          %async_token_9 = air.execute [%12] {
            memref.dealloc %results_8 : memref<64x64xbf16, 2>
          }
        }
        air.wait_all [%11, %async_token_5]  {air.segment_end}
      }
    }
    return
  }
}
