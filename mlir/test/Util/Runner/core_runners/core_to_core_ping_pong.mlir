//===- core_to_core_ping_pong.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/../arch.json -g core | FileCheck %s

// Core-to-core ping-pong buffering, running each core in a herd

// CHECK-COUNT-24: "name": "ChannelGetOp

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 == 0, s1 - 1 >= 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 - 2 >= 0, -s1 + 2 >= 0)>
module {
  air.channel @channel_3 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
      %1 = air.partition async  {
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async  tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c128 = arith.constant 128 : index
          %async_token, %results = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %async_token_1, %results_2 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %3 = affine.if #set()[%arg4, %arg5] -> !air.async.token {
            %4:3 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %async_token, %arg10 = %async_token_1, %arg11 = %async_token_1) -> (!air.async.token, !air.async.token, !air.async.token) {
              %5 = air.channel.put async [%arg9, %arg11]  @channel_1[] (%results[] [] []) : (memref<32x32xbf16, 2>)
              %6 = air.channel.put async [%5, %arg10]  @channel_1[] (%results_2[] [] []) : (memref<32x32xbf16, 2>)
              scf.yield %5, %6, %6 : !air.async.token, !air.async.token, !air.async.token
            }
            affine.yield %4#2 : !air.async.token
          } else {
            %4 = affine.if #set1()[%arg4, %arg5] -> !air.async.token {
              %5:4 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %async_token, %arg10 = %async_token_1, %arg11 = %async_token_1, %arg12 = %async_token_1) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
                %6 = air.channel.get async [%arg9, %arg12]  @channel_1[] (%results[] [] []) : (memref<32x32xbf16, 2>)
                %7 = air.channel.get async [%arg10, %6]  @channel_1[] (%results_2[] [] []) : (memref<32x32xbf16, 2>)
                %8 = air.channel.put async [%arg11, %6]  @channel_2[] (%results[] [] []) : (memref<32x32xbf16, 2>)
                %9 = air.channel.put async [%7, %8]  @channel_2[] (%results_2[] [] []) : (memref<32x32xbf16, 2>)
                scf.yield %8, %9, %9, %7 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
              }
              affine.yield %5#3 : !air.async.token
            } else {
              %5 = affine.if #set2()[%arg4, %arg5] -> !air.async.token {
                %6:4 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %async_token, %arg10 = %async_token_1, %arg11 = %async_token_1, %arg12 = %async_token_1) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
                  %7 = air.channel.get async [%arg9, %arg12]  @channel_2[] (%results[] [] []) : (memref<32x32xbf16, 2>)
                  %8 = air.channel.get async [%arg10, %7]  @channel_2[] (%results_2[] [] []) : (memref<32x32xbf16, 2>)
                  %9 = air.channel.put async [%arg11, %7]  @channel_3[] (%results[] [] []) : (memref<32x32xbf16, 2>)
                  %10 = air.channel.put async [%8, %9]  @channel_3[] (%results_2[] [] []) : (memref<32x32xbf16, 2>)
                  scf.yield %9, %10, %10, %8 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
                }
                affine.yield %6#3 : !air.async.token
              } else {
                %6:3 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %async_token, %arg10 = %async_token_1, %arg11 = %async_token_1) -> (!air.async.token, !air.async.token, !air.async.token) {
                  %7 = air.channel.get async [%arg9, %arg11]  @channel_3[] (%results[] [] []) : (memref<32x32xbf16, 2>)
                  %8 = air.channel.get async [%arg10, %7]  @channel_3[] (%results_2[] [] []) : (memref<32x32xbf16, 2>)
                  scf.yield %7, %8, %8 : !air.async.token, !air.async.token, !air.async.token
                }
                affine.yield %6#2 : !air.async.token
              }
              affine.yield %5 : !air.async.token
            }
            affine.yield %4 : !air.async.token
          }
          air.herd_terminator
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return
  }
}

