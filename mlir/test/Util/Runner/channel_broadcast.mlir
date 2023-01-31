//===- dma_broadcast.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Air channel ops with broadcast

// CHECK-COUNT-160: "name": "ChannelGetOp@channel

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
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
    %0 = air.launch async [%async_token_3] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%results_2) : memref<256x1024xbf16> {
      %c1_4 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async  {id = 8 : i32}
      %2 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %1) -> (!air.async.token) {
        %4 = scf.for %arg11 = %c0 to %c1024 step %c128 iter_args(%arg12 = %arg10) -> (!air.async.token) {
          %5 = air.channel.put async [%arg12]  @channel_4[] (%arg8[%arg9, %arg11] [%c128, %c128] [%c1024, %c1_4]) : (memref<256x1024xbf16>)
          scf.yield %5 : !air.async.token
        }
        scf.yield %4 : !air.async.token
      }
      %3 = air.partition async  {
        %c1_5 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_6 = arith.constant 0 : index
        %c1024_7 = arith.constant 1024 : index
        %c128_8 = arith.constant 128 : index
        %c256_9 = arith.constant 256 : index
        %c32 = arith.constant 32 : index
        %c96 = arith.constant 96 : index
        %c64 = arith.constant 64 : index
        %4 = air.wait_all async  {id = 8 : i32}
        %5 = scf.for %arg9 = %c0_6 to %c256_9 step %c128_8 iter_args(%arg10 = %4) -> (!air.async.token) {
          %6 = scf.for %arg11 = %c0_6 to %c1024_7 step %c128_8 iter_args(%arg12 = %arg10) -> (!air.async.token) {
            %async_token_10, %results_11 = air.execute [%arg12] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %7 = air.channel.get async [%async_token_10]  @channel_4[] (%results_11[] [] []) : (memref<128x128xbf16, 1>)
            %8 = air.channel.put async [%7]  @channel_0[] (%results_11[%c0_6, %c0_6] [%c32, %c32] [%c128_8, %c1_5]) : (memref<128x128xbf16, 1>)
            %9 = air.channel.put async [%7]  @channel_1[] (%results_11[%c32, %c0_6] [%c32, %c32] [%c128_8, %c1_5]) : (memref<128x128xbf16, 1>)
            %10 = air.channel.put async [%7]  @channel_2[] (%results_11[%c64, %c0_6] [%c32, %c32] [%c128_8, %c1_5]) : (memref<128x128xbf16, 1>)
            %11 = air.channel.put async [%7]  @channel_3[] (%results_11[%c96, %c0_6] [%c32, %c32] [%c128_8, %c1_5]) : (memref<128x128xbf16, 1>)
            %12 = air.herd @herd_0 async [%8, %10, %11, %9]  tile (%arg13, %arg14) in (%arg15=%c4, %arg16=%c4) {
              %async_token_12, %results_13 = air.execute -> (memref<32x32xbf16, 2>) {
                %alloc = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %alloc : memref<32x32xbf16, 2>
              }
              %13 = affine.if #set()[%arg13, %arg14] -> !air.async.token {
                %14 = air.channel.get async [%async_token_12]  @channel_0[%arg13, %arg14] (%results_13[] [] []) : (memref<32x32xbf16, 2>)
                affine.yield %14 : !air.async.token
              } else {
                %14 = affine.if #set1()[%arg13, %arg14] -> !air.async.token {
                  %15 = air.channel.get async [%async_token_12]  @channel_1[%arg13, %arg14] (%results_13[] [] []) : (memref<32x32xbf16, 2>)
                  affine.yield %15 : !air.async.token
                } else {
                  %15 = affine.if #set2()[%arg13, %arg14] -> !air.async.token {
                    %16 = air.channel.get async [%async_token_12]  @channel_2[%arg13, %arg14] (%results_13[] [] []) : (memref<32x32xbf16, 2>)
                    affine.yield %16 : !air.async.token
                  } else {
                    %16 = air.channel.get async [%async_token_12]  @channel_3[%arg13, %arg14] (%results_13[] [] []) : (memref<32x32xbf16, 2>)
                    affine.yield %16 : !air.async.token
                  }
                  affine.yield %15 : !air.async.token
                }
                affine.yield %14 : !air.async.token
              }
              air.herd_terminator
            }
            scf.yield %12 : !air.async.token
          }
          scf.yield %6 : !air.async.token
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

