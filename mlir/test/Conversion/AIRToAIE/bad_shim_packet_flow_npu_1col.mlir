//===- bad_shim_packet_flow_npu_1col.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(air-to-aie{row-offset=2 col-offset=0 device=npu1_1col use-pkt-flow-at-shim-dma=true})' --split-input-file -verify-diagnostics

// 4x4 NPU1 array. Should fail expectedly when allocating shim dma channels.

#map = affine_map<()[s0] -> (s0 * 256)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  air.channel @channel_12 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_13 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_14 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_15 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_2 [4, 1]
  func.func @func2(%arg0: memref<512x512xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0) : memref<512x512xbf16> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3]
        air.execute_terminator %3 : index
      }
      %1 = scf.for %arg8 = %c0 to %c512 step %c64 iter_args(%arg9 = %async_token) -> (!air.async.token) {
        %3 = air.channel.put async [%arg9]  @channel_2[%c0, %c0] (%arg7[%c0, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 1 : i32} : (memref<512x512xbf16>)
        %4 = air.channel.put async [%arg9]  @channel_2[%c1, %c0] (%arg7[%c1, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 2 : i32} : (memref<512x512xbf16>)
// expected-error@+1 {{'air.channel.put' op failed to map to shim dma channels: out of channels.}}
        %5 = air.channel.put async [%arg9]  @channel_2[%c2_0, %c0] (%arg7[%c2_0, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
        %6 = air.channel.put async [%arg9]  @channel_2[%c3, %c0] (%arg7[%c3, %c0, %results, %arg8] [%c1, %c1, %c64, %c64] [%c32768, %c64, %c512, %c1]) {id = 4 : i32} : (memref<512x512xbf16>)
        %7 = air.wait_all async [%3, %4, %5, %6] 
        scf.yield %7 : !air.async.token
      }
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c64_3 = arith.constant 64 : index
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %async_token_6, %results_7 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_8, %results_9 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %3 = air.wait_all async [%async_token_6, %async_token_8] 
        %4 = air.wait_all async [%async_token_6, %async_token_8] 
        %5:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %4, %arg10 = %async_token_8, %arg11 = %3, %arg12 = %3) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c0_4, %c0_4] (%results_7[] [] []) {id = 13 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_12[] (%results_7[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set, id = 14 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c0_4, %c0_4] (%results_9[] [] []) {id = 15 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_12[] (%results_9[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set, id = 16 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_10 = air.execute [%5#1] {
          memref.dealloc %results_7 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_11 = air.execute [%5#1] {
          memref.dealloc %results_9 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_12, %results_13 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_14, %results_15 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %6 = air.wait_all async [%async_token_12, %async_token_14] 
        %7 = air.wait_all async [%async_token_12, %async_token_14] 
        %8:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %7, %arg10 = %async_token_14, %arg11 = %6, %arg12 = %6) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c1_5, %c0_4] (%results_13[] [] []) {id = 17 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_13[] (%results_13[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set1, id = 18 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c1_5, %c0_4] (%results_15[] [] []) {id = 19 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_13[] (%results_15[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set1, id = 20 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_16 = air.execute [%8#1] {
          memref.dealloc %results_13 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%8#1] {
          memref.dealloc %results_15 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %9 = air.wait_all async [%async_token_18, %async_token_20] 
        %10 = air.wait_all async [%async_token_18, %async_token_20] 
        %11:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %10, %arg10 = %async_token_20, %arg11 = %9, %arg12 = %9) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c2_2, %c0_4] (%results_19[] [] []) {id = 21 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_14[] (%results_19[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set2, id = 22 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c2_2, %c0_4] (%results_21[] [] []) {id = 23 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_14[] (%results_21[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set2, id = 24 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_22 = air.execute [%11#1] {
          memref.dealloc %results_19 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_23 = air.execute [%11#1] {
          memref.dealloc %results_21 : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<1x1x64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x64xbf16, 1 : i32>
        }
        %12 = air.wait_all async [%async_token_24, %async_token_26] 
        %13 = air.wait_all async [%async_token_24, %async_token_26] 
        %14:4 = scf.for %arg8 = %c0_4 to %c8 step %c2_2 iter_args(%arg9 = %13, %arg10 = %async_token_26, %arg11 = %12, %arg12 = %12) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %16 = air.channel.get async [%arg9, %arg12]  @channel_2[%c3_1, %c0_4] (%results_25[] [] []) {id = 25 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %17 = air.channel.put async [%16, %arg11]  @channel_15[] (%results_25[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set3, id = 26 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %18 = air.channel.get async [%16, %arg10]  @channel_2[%c3_1, %c0_4] (%results_27[] [] []) {id = 27 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          %19 = air.channel.put async [%17, %arg9, %18]  @channel_15[] (%results_27[%c0_4, %c0_4, %c0_4] [%c8, %c64_3, %c8] [%c8, %c64_3, %c1_5]) {broadcast_set = #set3, id = 28 : i32} : (memref<1x1x64x64xbf16, 1 : i32>)
          scf.yield %17, %19, %19, %18 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %15 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 5 : i32, link_with = "mm.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token_28, %results_29 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
          }
          %16 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
            %17 = air.channel.get async [%async_token_28]  @channel_12[%arg8, %arg9] (%results_29[] [] []) {id = 61 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
            affine.yield %17 : !air.async.token
          } else {
            %17 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
              %18 = air.channel.get async [%async_token_28]  @channel_13[%arg8, %arg9] (%results_29[] [] []) {id = 62 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
              affine.yield %18 : !air.async.token
            } else {
              %18 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
                %19 = air.channel.get async [%async_token_28]  @channel_14[%arg8, %arg9] (%results_29[] [] []) {id = 63 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %19 : !air.async.token
              } else {
                %19 = air.channel.get async [%async_token_28]  @channel_15[%arg8, %arg9] (%results_29[] [] []) {id = 64 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %19 : !air.async.token
              }
              affine.yield %18 : !air.async.token
            }
            affine.yield %17 : !air.async.token
          }
        }
      }
    }
    return
  }
}
