//===- construct_ping_pong.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-construct-ping-pong-dependency-pattern | FileCheck %s

// Construct dependency edges in scf.for to represent ping-pong buffering
// CHECK-LABEL: channel_put_get
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_0[]
// CHECK: %[[EVENT6:.*]] = air.channel.put async [%[[EVENT3]], %[[EVENT5]]] @channel_1[]
// CHECK: %[[EVENT7:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_0[]
// CHECK: %[[EVENT8:.*]] = air.channel.put async [%[[EVENT6]], %[[EVENT7]]] @channel_1[]
// CHECK: scf.yield %[[EVENT6]], %[[EVENT8]], %[[EVENT8]], %[[EVENT7]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token

air.channel @channel_1 [1, 1]
air.channel @channel_0 [1, 1]
func.func @channel_put_get(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
    %1 = air.segment async  args(%arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7, %arg14=%arg8, %arg15=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %async_token, %results = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %2 = scf.for %arg16 = %c0 to %c512 step %c64 iter_args(%arg17 = %async_token) -> (!air.async.token) {
        %5 = air.channel.put async [%arg17]  @channel_0[] (%results[] [] []) : (memref<32x32xbf16, 1>)
        scf.yield %5 : !air.async.token
      }
      %3 = air.herd @herd_0 async [%async_token]  tile (%arg16, %arg17) in (%arg18=%c1_0, %arg19=%c1_0) {
        %c128 = arith.constant 128 : index
        %c0_2 = arith.constant 0 : index
        %c512_3 = arith.constant 512 : index
        %5 = air.wait_all async 
        %async_token_4, %results_5 = air.execute [%5] -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        } {unrolled_iteration = 1 : i32}
        %async_token_6, %results_7 = air.execute [%async_token_4] -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        } {unrolled_iteration = 0 : i32}
        %6 = scf.for %arg20 = %c0_2 to %c512_3 step %c128 iter_args(%arg21 = %async_token_6) -> (!air.async.token) {
          %7 = air.channel.get async [%arg21]  @channel_0[] (%results_7[] [] []) {async_front = true, unrolled_iteration = 0 : i32} : (memref<32x32xbf16, 2>)
          %8 = air.channel.put async [%7]  @channel_1[] (%results_7[] [] []) {async_back = true, unrolled_iteration = 0 : i32} : (memref<32x32xbf16, 2>)
          %9 = air.channel.get async [%8]  @channel_0[] (%results_5[] [] []) {async_front = true, unrolled_iteration = 1 : i32} : (memref<32x32xbf16, 2>)
          %10 = air.channel.put async [%9]  @channel_1[] (%results_5[] [] []) {async_back = true, unrolled_iteration = 1 : i32} : (memref<32x32xbf16, 2>)
          scf.yield %10 : !air.async.token
        } {unroll = 2 : i32}
        %async_token_8 = air.execute [%6] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        } {unrolled_iteration = 0 : i32}
        %async_token_9 = air.execute [%6] {
          memref.dealloc %results_5 : memref<32x32xbf16, 2>
        } {unrolled_iteration = 1 : i32}
        air.herd_terminator
      }
      %4 = scf.for %arg16 = %c0 to %c512 step %c64 iter_args(%arg17 = %async_token) -> (!air.async.token) {
        %5 = air.channel.get async [%arg17]  @channel_1[] (%results[] [] []) : (memref<32x32xbf16, 1>)
        scf.yield %5 : !air.async.token
      }
      %async_token_1 = air.execute [%4] {
        memref.dealloc %results : memref<32x32xbf16, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Check with the presence of AffineIfOp for broadcasting.
// CHECK-LABEL: affine_if
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = affine.if #set(){{.*}}-> !air.async.token {
// CHECK: %[[EVENT6:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_2
// CHECK: affine.yield %[[EVENT6]]
// CHECK: else
// CHECK: %[[EVENT7:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_3
// CHECK: affine.yield %[[EVENT7]]
// CHECK: %[[EVENT8:.*]] = air.channel.put async [%[[EVENT3]], %[[EVENT5]]] @channel_4
// CHECK: %[[EVENT9:.*]] = affine.if #set(){{.*}}-> !air.async.token {
// CHECK: %[[EVENT10:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_2
// CHECK: affine.yield %[[EVENT10]]
// CHECK: else
// CHECK: %[[EVENT11:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_3
// CHECK: affine.yield %[[EVENT11]]
// CHECK: %[[EVENT12:.*]] = air.channel.put async [%[[EVENT8]], %[[EVENT9]]] @channel_4
// CHECK: %[[EVENT13:.*]] = air.wait_all async [%[[EVENT9]]]
// CHECK: scf.yield %[[EVENT8]], %[[EVENT12]], %[[EVENT12]], %[[EVENT13]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
air.channel @channel_4 [1, 1]
air.channel @channel_3 [1, 1] {broadcast_shape = [1, 2]}
air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
func.func @affine_if(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
    %1 = air.segment async  args(%arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7, %arg14=%arg8, %arg15=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %async_token, %results = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %2 = scf.for %arg16 = %c0 to %c512 step %c64 iter_args(%arg17 = %async_token) -> (!air.async.token) {
        %5 = air.channel.put async [%arg17]  @channel_2[] (%results[] [] []) : (memref<32x32xbf16, 1>)
        scf.yield %5 : !air.async.token
      }
      %3 = air.herd @herd_0 async [%async_token]  tile (%arg16, %arg17) in (%arg18=%c2_0, %arg19=%c2_0) {
        %c128 = arith.constant 128 : index
        %c0_2 = arith.constant 0 : index
        %c512_3 = arith.constant 512 : index
        %5 = air.wait_all async 
        %async_token_4, %results_5 = air.execute [%5] -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        } {unrolled_iteration = 1 : i32}
        %async_token_6, %results_7 = air.execute [%async_token_4] -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        } {unrolled_iteration = 0 : i32}
        %6 = scf.for %arg20 = %c0_2 to %c512_3 step %c128 iter_args(%arg21 = %async_token_6) -> (!air.async.token) {
          %7 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %71 = air.channel.get async [%arg21]  @channel_2[%arg16, %arg17] (%results_7[] [] []) {async_front = true, unrolled_iteration = 0 : i32} : (memref<32x32xbf16, 2>)
            affine.yield %71 : !air.async.token
          } else {
            %71 = air.channel.get async [%arg21]  @channel_3[%arg16, %arg17] (%results_7[] [] []) {async_front = true, unrolled_iteration = 0 : i32} : (memref<32x32xbf16, 2>)
            affine.yield %71 : !air.async.token
          }
          %8 = air.channel.put async [%7]  @channel_4[] (%results_7[] [] []) {async_back = true, unrolled_iteration = 0 : i32} : (memref<32x32xbf16, 2>)
          %9 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %91 = air.channel.get async [%8]  @channel_2[%arg16, %arg17] (%results_5[] [] []) {async_front = true, unrolled_iteration = 1 : i32} : (memref<32x32xbf16, 2>)
            affine.yield %91 : !air.async.token
          } else {
            %91 = air.channel.get async [%8]  @channel_3[%arg16, %arg17] (%results_5[] [] []) {async_front = true, unrolled_iteration = 1 : i32} : (memref<32x32xbf16, 2>)
            affine.yield %91 : !air.async.token
          }
          %10 = air.channel.put async [%9]  @channel_4[] (%results_5[] [] []) {async_back = true, unrolled_iteration = 1 : i32} : (memref<32x32xbf16, 2>)
          scf.yield %10 : !air.async.token
        } {unroll = 2 : i32}
        %async_token_8 = air.execute [%6] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        } {unrolled_iteration = 0 : i32}
        %async_token_9 = air.execute [%6] {
          memref.dealloc %results_5 : memref<32x32xbf16, 2>
        } {unrolled_iteration = 1 : i32}
        air.herd_terminator
      }
      %4 = scf.for %arg16 = %c0 to %c512 step %c64 iter_args(%arg17 = %async_token) -> (!air.async.token) {
        %5 = air.channel.get async [%arg17]  @channel_4[] (%results[] [] []) : (memref<32x32xbf16, 1>)
        scf.yield %5 : !air.async.token
      }
      %async_token_1 = air.execute [%4] {
        memref.dealloc %results : memref<32x32xbf16, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
