//===- two_by_two_herd_dataflow.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/../arch.json -g core | FileCheck %s

// Air channels forming dataflow through a two-by-two herd. 

// CHECK: "name": "ChannelPutOp@channel_0(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0:.*]],

// CHECK: "name": "ChannelPutOp@channel_0(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0]],

// CHECK: "name": "ChannelPutOp@channel_0(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME1:.*]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME1]],

// CHECK: "name": "ChannelPutOp@channel_0(L1-->L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME1]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME1]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME2:.*]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L1)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": [[TIME2]],


// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 2 >= 0)>
module {
  air.channel @channel_0 [1, 2]
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
      %1 = air.partition async  {
        %c2 = arith.constant 2 : index
        %async_token_4, %results_5 = air.execute -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %2 = air.herd @herd_0 async [%async_token_4]  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) {
          %c0 = arith.constant 0 : index
          %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %3 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
            %4 = air.channel.put async [%async_token_6]  @channel_0[%c0, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %4 : !air.async.token
          } else {
            %4 = air.channel.get async [%async_token_6]  @channel_0[%c0, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %4 : !air.async.token
          }
          %async_token_8 = air.execute [%3] {
            memref.dealloc %results_7 : memref<32x32xbf16, 2>
          }
          air.herd_terminator
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

