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
      }
      %7 = scf.for %arg10 = %c0_1 to %c512_1 step %c64_1 iter_args(%arg11 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.get async [%arg11]  @channel_1[] (%results_2[][][]) : (memref<32x32xbf16, 1>)
        scf.yield %8 : !air.async.token
      }
      %async_token_2 = air.execute [%7] {
        memref.dealloc %results_2 : memref<32x32xbf16, 1>
      }
    }
  }
  return
}

// Label async_front based on tokens declared outside of for loop.
// CHECK-LABEL: test1
// CHECK: air.segment
// CHECK: air.wait_all async
// CHECK: air.wait_all async
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}async_front = true
// CHECK: air.channel.get{{.*}}async_front = true
// CHECK: air.wait_all async{{.*}}async_back = true

func.func @test1(%arg0: memref<2048xi8>, %arg1: memref<2048x1024xi8>, %arg2: memref<1024xi32>) {
  %c4 = arith.constant 4 : index
  %0 = air.launch async (%arg3) in (%arg4=%c4) {
    %1 = air.segment @vecmat_i8_0 async  {
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2048 = arith.constant 2048 : index
      %c128 = arith.constant 128 : index
      %2 = air.wait_all async 
      %3 = air.wait_all async 
      %11 = scf.for %arg5 = %c0 to %c2048 step %c128 iter_args(%arg9 = %3) -> (!air.async.token) {
        %async_token, %results = air.execute -> (memref<128xi8, 1>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<128xi8, 1>
          air.execute_terminator %alloc : memref<128xi8, 1>
        }
        %async_token_0, %results_1 = air.execute -> (memref<128x256xi8, 1>) {
          %alloc = memref.alloc() {hoist_alloc = true} : memref<128x256xi8, 1>
          air.execute_terminator %alloc : memref<128x256xi8, 1>
        }
        %4 = air.channel.get async [%2]  @channel_1[] (%results[%arg5] [%c128] [%c1]) {id = 4 : i32} : (memref<128xi8, 1>)
        %5 = air.channel.get async [%3]  @channel_2[] (%results_1[%arg5, %c0] [%c128, %c256] [%c256, %c1]) {id = 5 : i32} : (memref<128x256xi8, 1>)
        %6 = air.channel.put async [%4]  @channel_0[] (%results[%c0, %arg5] [%c8, %c16] [%c16, %c1]) {id = 6 : i32} : (memref<128xi8, 1>)
        %7 = air.channel.put async [%5]  @channel_3[%c0, %c0] (%results_1[%c0, %c0, %arg5, %c0] [%c16, %c8, %c16, %c8] [%c8, %c4096, %c256, %c1]) {id = 7 : i32} : (memref<128x256xi8, 1>)
        %8 = air.channel.put async [%5]  @channel_3[%c1, %c0] (%results_1[%c0, %c0, %arg5, %c128] [%c16, %c8, %c16, %c8] [%c8, %c4096, %c256, %c1]) {id = 7 : i32} : (memref<128x256xi8, 1>)
        %async_token_2 = air.execute {
          memref.dealloc %results : memref<128xi8, 1>
        }
        %async_token_3 = air.execute {
          memref.dealloc %results_1 : memref<128x256xi8, 1>
        }
        %9 = air.wait_all async [%async_token, %async_token_0, %4, %5, %6, %7, %8, %async_token_2, %async_token_3] 
        scf.yield %9 : !air.async.token
      } {isolated = true, unroll = 2 : i32}
    }
  }
  return
}
