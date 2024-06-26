//===- data_movement.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Test trace latency modelling of data movement.

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,TIME0:]],

// CHECK: "name": "ChannelGetOp@channel_2(L1<--L2)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,TIME1:]],

// CHECK: "name": "ChannelGetOp@channel_2(L1<--L2)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#TIME1 + 512]],

// CHECK: "name": "ChannelGetOp@channel_4(L1<--L2)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.[[#%d,TIME2:]],

// CHECK: "name": "ChannelGetOp@channel_4(L1<--L2)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#TIME2 + 256]],

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK: "ph": "E",
// CHECK: "ts": 1.0[[#TIME0 + 1024 - 1000]],

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_5 [1, 1]
  func.func @test(%arg0: memref<32x32xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>, %arg3: memref<1024x1024xi32>) -> memref<32x32xi32> {
    %c1 = arith.constant 1 : index
    %async_token_1, %results_2 = air.execute -> (memref<32x32xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xi32>
      air.execute_terminator %alloc : memref<32x32xi32>
    }
    %0 = air.launch async [%async_token_1] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<32x32xi32>, memref<1024x1024xi32> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<32x32xi32>, memref<1024x1024xi32> attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c4 = arith.constant 4 : index
        %c1_0 = arith.constant 1 : index
        %async_token_3, %results_4 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %3 = air.channel.put async [%async_token_3]  @channel_0[] (%results_4[] [] []) : (memref<32x32xi32, 1>)
        %async_token_5, %results_6 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %4 = air.channel.get async [%async_token_5]  @channel_1[] (%results_6[] [] []) : (memref<32x32xi32, 1>)
        %async_token_7, %results_8 = air.execute -> (memref<32x32xbf16, 1>) {
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xbf16, 1>
          air.execute_terminator %alloc : memref<32x32xbf16, 1>
        }
        %5 = air.channel.put async [%async_token_7]  @channel_2[] (%results_8[] [] []) : (memref<32x32xbf16, 1>)
        %async_token_9, %results_10 = air.execute -> (memref<32x32xbf16, 1>) {
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xbf16, 1>
          air.execute_terminator %alloc : memref<32x32xbf16, 1>
        }
        %6 = air.channel.get async [%async_token_9]  @channel_3[] (%results_10[] [] []) : (memref<32x32xbf16, 1>)
        %async_token_11, %results_12 = air.execute -> (memref<32x32xi8, 1>) {
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xi8, 1>
          air.execute_terminator %alloc : memref<32x32xi8, 1>
        }
        %7 = air.channel.put async [%async_token_11]  @channel_4[] (%results_12[] [] []) : (memref<32x32xi8, 1>)
        %async_token_13, %results_14 = air.execute -> (memref<32x32xi8, 1>) {
          %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xi8, 1>
          air.execute_terminator %alloc : memref<32x32xi8, 1>
        }
        %8 = air.channel.get async [%async_token_13]  @channel_5[] (%results_14[] [] []) : (memref<32x32xi8, 1>)
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c1_0, %arg24=%c1_0) {
          %cst_8 = arith.constant 2 : i32
          %cst_9 = arith.constant 1 : i32
          %cst_10 = arith.constant 1 : i32
          %async_token_15, %results_16 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %9 = air.channel.get async [%async_token_15]  @channel_0[] (%results_16[] [] []) : (memref<32x32xi32, 2>)
          %10 = air.channel.put async [%9]  @channel_1[] (%results_16[] [] []) : (memref<32x32xi32, 2>)
          %async_token_17 = air.execute [%10] {
            memref.dealloc %results_16 : memref<32x32xi32, 2>
          }
          %async_token_19, %results_20 = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %11 = air.channel.get async [%async_token_19]  @channel_2[] (%results_20[] [] []) : (memref<32x32xbf16, 2>)
          %12 = air.channel.put async [%11]  @channel_3[] (%results_20[] [] []) : (memref<32x32xbf16, 2>)
          %async_token_21 = air.execute [%12] {
            memref.dealloc %results_20 : memref<32x32xbf16, 2>
          }
          %async_token_23, %results_24 = air.execute -> (memref<32x32xi8, 2>) {
            %alloc = memref.alloc() : memref<32x32xi8, 2>
            air.execute_terminator %alloc : memref<32x32xi8, 2>
          }
          %13 = air.channel.get async [%async_token_23]  @channel_4[] (%results_24[] [] []) : (memref<32x32xi8, 2>)
          %14 = air.channel.put async [%13]  @channel_5[] (%results_24[] [] []) : (memref<32x32xi8, 2>)
          %async_token_25 = air.execute [%14] {
            memref.dealloc %results_24 : memref<32x32xi8, 2>
          }
        }
      }
    }
    return %results_2 : memref<32x32xi32>
  }
}
