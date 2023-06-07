//===- segment_dataflow.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// A dataflow pipeline made of air.segments as pipeline stages.
// Stages are connected with FIFOs to enable concurrent execution.

// CHECK: "name": "ChannelGetOp@channel_5(L2<--L2)",
// CHECK: "name": "ChannelGetOp@channel_1(L2<--L3)",
// CHECK: "name": "ChannelGetOp@channel_9(L2<--L2)",
// CHECK: "name": "ChannelGetOp@channel_5(L2<--L2)",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
  air.channel @channel_11 [1, 1]
  air.channel @channel_10 [1, 1]
  air.channel @channel_9 [1, 1]
  air.channel @channel_8 [1, 1]
  air.channel @channel_7 [1, 1]
  air.channel @channel_6 [1, 1]
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @test(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) -> memref<1024x1024xbf16> {
    %c1 = arith.constant 1 : index
    %async_token, %results = air.execute -> (memref<1024x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xbf16>
      air.execute_terminator %alloc : memref<1024x1024xbf16>
    }
    %async_token_0, %results_1 = air.execute -> (memref<1024x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xbf16>
      air.execute_terminator %alloc : memref<1024x1024xbf16>
    }
    %async_token_2, %results_3 = air.execute -> (memref<1024x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xbf16>
      air.execute_terminator %alloc : memref<1024x1024xbf16>
    }
    %async_token_4, %results_5 = air.execute -> (memref<1024x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xbf16>
      air.execute_terminator %alloc : memref<1024x1024xbf16>
    }
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%results_1, %arg11=%arg2, %arg12=%arg3, %arg13=%results_3, %arg14=%results_5) : memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16>, memref<1024x1024xbf16> attributes {id = 1 : i32} {
      %c1_6 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg15 = %c0 to %c1024 step %c256 iter_args(%arg16 = %1) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_0[] (%arg10[%arg15, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 1 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %3 = air.wait_all async 
      %4 = scf.for %arg15 = %c0 to %c1024 step %c128 iter_args(%arg16 = %3) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_1[] (%arg8[%arg15, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 2 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %5 = air.wait_all async 
      %6 = scf.for %arg15 = %c0 to %c1024 step %c128 iter_args(%arg16 = %5) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_2[] (%arg9[%c0, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 3 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %7 = air.segment async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c0_7 = arith.constant 0 : index
        %c1024_8 = arith.constant 1024 : index
        %c128_9 = arith.constant 128 : index
        %c256_10 = arith.constant 256 : index
        %20 = air.wait_all async 
        %21 = scf.for %arg15 = %c0_7 to %c1024_8 step %c256_10 iter_args(%arg16 = %20) -> (!air.async.token) {
          %async_token_11, %results_12 = air.execute [%arg16] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %22 = air.channel.get async [%async_token_11]  @channel_0[] (%results_12[] [] []) {id = 5 : i32} : (memref<128x128xbf16, 1>)
          %23 = scf.for %arg17 = %c0_7 to %c256_10 step %c128_9 iter_args(%arg18 = %22) -> (!air.async.token) {
            %async_token_14, %results_15 = air.execute [%arg18] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %async_token_16, %results_17 = air.execute [%arg18] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %25 = air.channel.get async [%async_token_14]  @channel_1[] (%results_15[] [] []) {id = 6 : i32} : (memref<128x128xbf16, 1>)
            %26 = air.channel.get async [%async_token_16]  @channel_2[] (%results_17[] [] []) {id = 7 : i32} : (memref<128x128xbf16, 1>)
            %async_token_18 = air.execute [%25] {
              memref.dealloc %results_15 : memref<128x128xbf16, 1>
            }
            %async_token_19 = air.execute [%26] {
              memref.dealloc %results_17 : memref<128x128xbf16, 1>
            }
            %27 = air.wait_all async [%async_token_18, %async_token_19] 
            scf.yield %27 : !air.async.token
          }
          %24 = air.channel.put async [%23]  @channel_3[] (%results_12[] [] []) {id = 16 : i32} : (memref<128x128xbf16, 1>)
          %async_token_13 = air.execute [%24] {
            memref.dealloc %results_12 : memref<128x128xbf16, 1>
          }
          scf.yield %async_token_13 : !air.async.token
        }
        air.segment_terminator
      }
      %8 = air.wait_all async 
      %9 = scf.for %arg15 = %c0 to %c1024 step %c256 iter_args(%arg16 = %8) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_4[] (%arg13[%arg15, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 17 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %10 = air.wait_all async 
      %11 = scf.for %arg15 = %c0 to %c1024 step %c128 iter_args(%arg16 = %10) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_6[] (%arg11[%c0, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 19 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %12 = air.segment async  attributes {id = 4 : i32, x_loc = 4 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c1_7 = arith.constant 1 : index
        %c0_8 = arith.constant 0 : index
        %c1024_9 = arith.constant 1024 : index
        %c128_10 = arith.constant 128 : index
        %c256_11 = arith.constant 256 : index
        %async_token_12, %results_13 = air.execute -> (memref<128x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x1024xbf16, 1>
          air.execute_terminator %alloc : memref<128x1024xbf16, 1>
        }
        %20 = scf.for %arg15 = %c0_8 to %c1024_9 step %c256_11 iter_args(%arg16 = %async_token_12) -> (!air.async.token) {
          %23 = air.channel.get async [%arg16]  @channel_3[] (%results_13[%arg15, %c0_8] [%c128_10, %c128_10] [%c1024_9, %c1_7]) {id = 4 : i32} : (memref<128x1024xbf16, 1>)
          %24 = scf.for %arg17 = %c0_8 to %c256_11 step %c128_10 iter_args(%arg18 = %23) -> (!air.async.token) {
            %25 = air.channel.put async [%arg18]  @channel_5[] (%results_13[%c0_8, %c0_8] [%c128_10, %c128_10] [%c128_10, %c1_7]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
            scf.yield %25 : !air.async.token
          }
          scf.yield %24 : !air.async.token
        }
        %async_token_14 = air.execute [%20] {
          memref.dealloc %results_13 : memref<128x1024xbf16, 1>
        }
        %21 = air.wait_all async 
        %22 = scf.for %arg15 = %c0_8 to %c1024_9 step %c256_11 iter_args(%arg16 = %21) -> (!air.async.token) {
          %async_token_15, %results_16 = air.execute [%arg16] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %23 = air.channel.get async [%async_token_15]  @channel_4[] (%results_16[] [] []) {id = 21 : i32} : (memref<128x128xbf16, 1>)
          %24 = scf.for %arg17 = %c0_8 to %c256_11 step %c128_10 iter_args(%arg18 = %23) -> (!air.async.token) {
            %async_token_18, %results_19 = air.execute [%arg18] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %async_token_20, %results_21 = air.execute [%arg18] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %26 = air.channel.get async [%async_token_18]  @channel_5[] (%results_19[] [] []) {id = 22 : i32} : (memref<128x128xbf16, 1>)
            %27 = air.channel.get async [%async_token_20]  @channel_6[] (%results_21[] [] []) {id = 23 : i32} : (memref<128x128xbf16, 1>)
            %async_token_22 = air.execute [%26] {
              memref.dealloc %results_19 : memref<128x128xbf16, 1>
            }
            %async_token_23 = air.execute [%27] {
              memref.dealloc %results_21 : memref<128x128xbf16, 1>
            }
            %28 = air.wait_all async [%async_token_23, %async_token_22] 
            scf.yield %28 : !air.async.token
          }
          %25 = air.channel.put async [%24]  @channel_7[] (%results_16[] [] []) {id = 32 : i32} : (memref<128x128xbf16, 1>)
          %async_token_17 = air.execute [%25] {
            memref.dealloc %results_16 : memref<128x128xbf16, 1>
          }
          scf.yield %async_token_17 : !air.async.token
        }
        air.segment_terminator
      }
      %13 = air.wait_all async 
      %14 = scf.for %arg15 = %c0 to %c1024 step %c256 iter_args(%arg16 = %13) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_8[] (%arg14[%arg15, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 33 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %15 = air.wait_all async 
      %16 = scf.for %arg15 = %c0 to %c1024 step %c128 iter_args(%arg16 = %15) -> (!air.async.token) {
        %20 = air.channel.put async [%arg16]  @channel_10[] (%arg12[%c0, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 35 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %17 = air.wait_all async 
      %18 = scf.for %arg15 = %c0 to %c1024 step %c256 iter_args(%arg16 = %17) -> (!air.async.token) {
        %20 = air.channel.get async [%arg16]  @channel_11[] (%arg14[%arg15, %c0] [%c128, %c128] [%c1024, %c1_6]) {id = 36 : i32} : (memref<1024x1024xbf16>)
        scf.yield %20 : !air.async.token
      }
      %19 = air.segment async  attributes {id = 6 : i32, x_loc = 8 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c1_7 = arith.constant 1 : index
        %c0_8 = arith.constant 0 : index
        %c1024_9 = arith.constant 1024 : index
        %c128_10 = arith.constant 128 : index
        %c256_11 = arith.constant 256 : index
        %async_token_12, %results_13 = air.execute -> (memref<128x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x1024xbf16, 1>
          air.execute_terminator %alloc : memref<128x1024xbf16, 1>
        }
        %20 = scf.for %arg15 = %c0_8 to %c1024_9 step %c256_11 iter_args(%arg16 = %async_token_12) -> (!air.async.token) {
          %23 = air.channel.get async [%arg16]  @channel_7[] (%results_13[%arg15, %c0_8] [%c128_10, %c128_10] [%c1024_9, %c1_7]) {id = 4 : i32} : (memref<128x1024xbf16, 1>)
          %24 = scf.for %arg17 = %c0_8 to %c256_11 step %c128_10 iter_args(%arg18 = %23) -> (!air.async.token) {
            %25 = air.channel.put async [%arg18]  @channel_9[] (%results_13[%arg15, %c0_8] [%c128_10, %c128_10] [%c128_10, %c1_7]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
            scf.yield %25 : !air.async.token
          }
          scf.yield %24 : !air.async.token
        }
        %async_token_14 = air.execute [%20] {
          memref.dealloc %results_13 : memref<128x1024xbf16, 1>
        }
        %21 = air.wait_all async 
        %22 = scf.for %arg15 = %c0_8 to %c1024_9 step %c256_11 iter_args(%arg16 = %21) -> (!air.async.token) {
          %async_token_15, %results_16 = air.execute [%arg16] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %23 = air.channel.get async [%async_token_15]  @channel_8[] (%results_16[] [] []) {id = 37 : i32} : (memref<128x128xbf16, 1>)
          %24 = scf.for %arg17 = %c0_8 to %c256_11 step %c128_10 iter_args(%arg18 = %23) -> (!air.async.token) {
            %async_token_18, %results_19 = air.execute [%arg18] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %async_token_20, %results_21 = air.execute [%arg18] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %26 = air.channel.get async [%async_token_18]  @channel_9[] (%results_19[] [] []) {id = 38 : i32} : (memref<128x128xbf16, 1>)
            %27 = air.channel.get async [%async_token_20]  @channel_10[] (%results_21[] [] []) {id = 39 : i32} : (memref<128x128xbf16, 1>)
            %async_token_22 = air.execute [%26] {
              memref.dealloc %results_19 : memref<128x128xbf16, 1>
            }
            %async_token_23 = air.execute [%27] {
              memref.dealloc %results_21 : memref<128x128xbf16, 1>
            }
            %28 = air.wait_all async [%async_token_23, %async_token_22] 
            scf.yield %28 : !air.async.token
          }
          %25 = air.channel.put async [%24]  @channel_11[] (%results_16[] [] []) {id = 48 : i32} : (memref<128x128xbf16, 1>)
          %async_token_17 = air.execute [%25] {
            memref.dealloc %results_16 : memref<128x128xbf16, 1>
          }
          scf.yield %async_token_17 : !air.async.token
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %results_5 : memref<1024x1024xbf16>
  }
}
