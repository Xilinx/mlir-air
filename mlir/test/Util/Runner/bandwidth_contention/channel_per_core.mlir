//===- channel_per_core.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json -g core | FileCheck %s

// Check for correct event serialization with bandwidth contention


// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0:.*]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME0]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME1:.*]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME1]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME1]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME1]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME2:.*]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME2]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME2]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME2]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME3:.*]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME3]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME3]],

// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": [[TIME3]],


// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_1 [4, 4]
  air.channel @channel_0 [1, 1]
  func.func @test(%arg0: memref<128x128xbf16>, %arg1: memref<1024x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0) : memref<128x128xbf16> {
      %c1_4 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %1 = air.channel.put async  @channel_0[] (%arg8[] [] []) : (memref<128x128xbf16>)
      %3 = air.segment async attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c32 = arith.constant 32 : index
        %c1_5 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_6 = arith.constant 0 : index
        %c1024_7 = arith.constant 1024 : index
        %c128_8 = arith.constant 128 : index
        %c256_9 = arith.constant 256 : index
        %async_token_10, %results_11 = air.execute -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %4 = air.channel.get async [%async_token_10]  @channel_0[] (%results_11[] [] []) : (memref<128x128xbf16, 1>)
        %5 = scf.parallel (%arg15, %arg16) = (%c0_6, %c0_6) to (%c4, %c4) step (%c1_5, %c1_5) init (%4) -> !air.async.token {
          %async_token_18, %results_19 = air.execute [%4] -> (index) {
            %13 = affine.apply #map()[%arg15]
            air.execute_terminator %13 : index
          }
          %async_token_20, %results_21 = air.execute [%4] -> (index) {
            %13 = affine.apply #map()[%arg16]
            air.execute_terminator %13 : index
          }
          %12 = air.channel.put async [%async_token_20, %async_token_18]  @channel_1[%arg15, %arg16] (%results_11[%results_19, %results_21] [%c32, %c32] [%c128_8, %c1_5]) : (memref<128x128xbf16, 1>)
          scf.reduce(%12)  : !air.async.token {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %13 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %13 : !air.async.token
          }
          scf.yield
        }
        %10 = air.herd @herd_0 async tile (%arg15, %arg16) in (%arg17=%c4, %arg18=%c4) {
          %async_token_18, %results_19 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %13 = air.channel.get async [%async_token_18]  @channel_1[%arg15, %arg16] (%results_19[] [] []) : (memref<32x32xbf16, 2>)
          %async_token_22 = air.execute [%13] {
            memref.dealloc %results_19 : memref<32x32xbf16, 2>
          }
          air.herd_terminator
        }
        %async_token_23 = air.execute [%4] {
          memref.dealloc %results_11 : memref<128x128xbf16, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

