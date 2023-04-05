//===- one_by_four_herd_dataflow.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/../arch.json -g core | FileCheck %s

// Air channel forming a dataflow through a one-by-four herd. 


// CHECK: "name": "ChannelPutOp@channel_0(L2-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0:.*]],

// CHECK: "name": "ChannelPutOp@channel_0(L2-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME1:.*]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME2:.*]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME3:.*]],

// CHECK: "name": "ChannelPutOp@channel_1(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME3]],

// CHECK: "name": "ChannelPutOp@channel_1(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME4:.*]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME4]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME5:.*]],

// CHECK: "name": "ChannelPutOp@channel_2(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME5]],

// CHECK: "name": "ChannelPutOp@channel_2(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME6:.*]],

// CHECK: "name": "ChannelGetOp@channel_2(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME6]],

// CHECK: "name": "ChannelGetOp@channel_2(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME7:.*]],

// CHECK: "name": "ChannelPutOp@channel_3(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME7]],

// CHECK: "name": "ChannelPutOp@channel_3(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME8:.*]],

// CHECK: "name": "ChannelGetOp@channel_3(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME8]],

// CHECK: "name": "ChannelGetOp@channel_3(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME9:.*]],

// CHECK: "name": "ChannelPutOp@channel_4(L1-->L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME9]],

// CHECK: "name": "ChannelPutOp@channel_4(L1-->L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME10:.*]],

// CHECK: "name": "ChannelGetOp@channel_4(L2<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME11:.*]],

// CHECK: "name": "ChannelGetOp@channel_4(L2<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME12:.*]],

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 == 0, s1 - 1 >= 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 - 2 >= 0, -s1 + 2 >= 0)>
module {
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) -> memref<256x1024xbf16> {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %async_token, %results = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    } {id = 1 : i32}
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%cst : bf16) outs(%results : memref<256x1024xbf16>)
    } {id = 2 : i32}
    %async_token_1, %results_2 = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    } {id = 3 : i32}
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<256x1024xbf16> to memref<256x1024xbf16>
    } {id = 4 : i32}
    %0 = air.launch async [%async_token_3] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
      %1 = air.segment async  {
        %c1_4 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %async_token_5, %results_6 = air.execute -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %2 = air.channel.put async [%async_token_5]  @channel_0[] (%results_6[%c0, %c0] [%c32, %c32] [%c128, %c1_4]) : (memref<128x128xbf16, 1>)
        %3 = air.herd @herd_0 async [%2]  tile (%arg8, %arg9) in (%arg10=%c1_4, %arg11=%c4) {
          %async_token_7, %results_8 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %5 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
            %6 = air.channel.get async [%async_token_7]  @channel_0[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
            %7 = air.channel.put async [%6]  @channel_1[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %7 : !air.async.token
          } else {
            %6 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
              %7 = air.channel.get async [%async_token_7]  @channel_1[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
              %8 = air.channel.put async [%7]  @channel_2[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
              affine.yield %8 : !air.async.token
            } else {
              %7 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
                %8 = air.channel.get async [%async_token_7]  @channel_2[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
                %9 = air.channel.put async [%8]  @channel_3[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
                affine.yield %9 : !air.async.token
              } else {
                %8 = air.channel.get async [%async_token_7]  @channel_3[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
                %9 = air.channel.put async [%8]  @channel_4[] (%results_8[] [] []) : (memref<32x32xbf16, 2>)
                affine.yield %9 : !air.async.token
              }
              affine.yield %7 : !air.async.token
            }
            affine.yield %6 : !air.async.token
          }
          air.herd_terminator
        }
        %4 = air.channel.get async [%3]  @channel_4[] (%results_6[%c0, %c0] [%c32, %c32] [%c128, %c1_4]) : (memref<128x128xbf16, 1>)
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

