//===- annotate_front_back.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-front-and-back-ops-in-for-pattern | FileCheck %s

// Annotate fronts and backs in the dependency graph of a scf.for loop.
// CHECK-LABEL: test
// CHECK: air.herd
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}async_front = true
// CHECK: air.channel.put{{.*}}async_back = true

air.channel @channel_1 [1, 1]
air.channel @channel_0 [1, 1]
func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
    %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
      %c4 = arith.constant 4 : index
      %c1_1 = arith.constant 1 : index
      %c0_1 = arith.constant 0 : index
      %c512_1 = arith.constant 512: index
      %c64_1 = arith.constant 64 : index
      %async_token_1, %results_2 = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %4 = scf.for %arg10 = %c0_1 to %c512_1 step %c64_1 iter_args(%arg11 = %async_token_1) -> (!air.async.token) {
        %5 = air.channel.put async [%arg11]  @channel_0[] (%results_2[][][]) : (memref<32x32xbf16, 1>)
        scf.yield %5 : !air.async.token
      }
      %2 = air.herd @herd_0 async [%async_token_1] tile (%arg21, %arg22) in (%arg23=%c1_1, %arg24=%c1_1) {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %async_token_0 = air.wait_all async
        %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
          %async_token_3, %results_4 = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %5 = air.channel.get async [%async_token_3]  @channel_0[] (%results_4[][][]) : (memref<32x32xbf16, 2>)
          %6 = air.channel.put async [%5]  @channel_1[] (%results_4[][][]) : (memref<32x32xbf16, 2>)
          %async_token_5 = air.execute [%5] {
            memref.dealloc %results_4 : memref<32x32xbf16, 2>
          }
          scf.yield %6 : !air.async.token
        } {unroll = 2 : i32}
        air.herd_terminator
      }
      %7 = scf.for %arg10 = %c0_1 to %c512_1 step %c64_1 iter_args(%arg11 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.get async [%arg11]  @channel_1[] (%results_2[][][]) : (memref<32x32xbf16, 1>)
        scf.yield %8 : !air.async.token
      }
      %async_token_2 = air.execute [%7] {
        memref.dealloc %results_2 : memref<32x32xbf16, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
