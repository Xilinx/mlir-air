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

// Check with the presence of scf.for loop as consumer thread.
// CHECK-LABEL: scf_for
// CHECK: air.segment async
// CHECK: air.segment async
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_5
// CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT3]], %[[EVENT5]]]
// CHECK: %[[EVENT7:.*]] = scf.for
// CHECK: %[[EVENT8:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_5
// CHECK-NEXT: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT8]], %[[EVENT9]]]
// CHECK-NEXT: %[[EVENT11:.*]] = scf.for
// CHECK: %[[EVENT12:.*]] = air.wait_all async [%[[EVENT11]]]
// CHECK: scf.yield %[[EVENT8]], %[[EVENT12]], %[[EVENT12]], %[[EVENT9]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token
// CHECK: air.segment async
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_7
// CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT3]], %[[EVENT5]]]
// CHECK: %[[EVENT7:.*]] = scf.for
// CHECK: %[[EVENT8:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_7
// CHECK-NEXT: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT8]], %[[EVENT9]]]
// CHECK-NEXT: %[[EVENT11:.*]] = scf.for
// CHECK: %[[EVENT12:.*]] = air.wait_all async [%[[EVENT11]]]
// CHECK: scf.yield %[[EVENT8]], %[[EVENT12]], %[[EVENT12]], %[[EVENT9]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token

air.channel @channel_8 [1, 1]
air.channel @channel_7 [1, 1]
air.channel @channel_6 [1, 1]
air.channel @channel_5 [1, 1]
func.func @scf_for() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
    %1 = air.segment async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %4 = air.wait_all async 
      %5 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %4) -> (!air.async.token) {
        %async_token, %results = air.execute [%arg5] -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %6 = air.channel.put async [%async_token]  @channel_5[] (%results[] [] []) {id = 16 : i32} : (memref<128x128xbf16, 1>)
        %async_token_0 = air.execute [%6] {
          memref.dealloc %results : memref<128x128xbf16, 1>
        }
        scf.yield %async_token_0 : !air.async.token
      }
      air.segment_terminator
    }
    %2 = air.segment async  attributes {id = 4 : i32, x_loc = 4 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
      %c512 = arith.constant 512 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %4 = air.wait_all async 
      %async_token, %results = air.execute [%4] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %async_token_1, %results_2 = air.execute [%async_token] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %5 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.get async [%arg5]  @channel_5[] (%results_2[%arg4, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 0 : i32} : (memref<128x1024xbf16, 1>)
        %9 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %8) -> (!air.async.token) {
          %15 = air.channel.put async [%arg7]  @channel_6[] (%results_2[%c0, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.yield %15 : !air.async.token
        } {unrolled_iteration = 0 : i32}
        %10 = air.wait_all async [%9]  {async_back = true, unrolled_iteration = 0 : i32}
        %11 = arith.addi %arg4, %c256 : index
        %12 = air.channel.get async [%10]  @channel_5[] (%results[%11, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 1 : i32} : (memref<128x1024xbf16, 1>)
        %13 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %12) -> (!air.async.token) {
          %15 = air.channel.put async [%arg7]  @channel_6[] (%results[%c0, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.yield %15 : !air.async.token
        } {unrolled_iteration = 1 : i32}
        %14 = air.wait_all async [%13]  {async_back = true, unrolled_iteration = 1 : i32}
        scf.yield %14 : !air.async.token
      } {unroll = 2 : i64}
      %async_token_3 = air.execute [%5] {
        memref.dealloc %results_2 : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %async_token_4 = air.execute [%5] {
        memref.dealloc %results : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %6 = air.wait_all async 
      %7 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %6) -> (!air.async.token) {
        %async_token_5, %results_6 = air.execute [%arg5] -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %8 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg7] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %10 = air.channel.get async [%async_token_8]  @channel_6[] (%results_9[] [] []) {id = 22 : i32} : (memref<128x128xbf16, 1>)
          %async_token_10 = air.execute [%10] {
            memref.dealloc %results_9 : memref<128x128xbf16, 1>
          }
          %11 = air.wait_all async [%async_token_10] 
          scf.yield %11 : !air.async.token
        }
        %9 = air.channel.put async [%8]  @channel_7[] (%results_6[] [] []) {id = 32 : i32} : (memref<128x128xbf16, 1>)
        %async_token_7 = air.execute [%9] {
          memref.dealloc %results_6 : memref<128x128xbf16, 1>
        }
        scf.yield %async_token_7 : !air.async.token
      }
      air.segment_terminator
    }
    %3 = air.segment async  attributes {id = 6 : i32, x_loc = 8 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
      %c512 = arith.constant 512 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %4 = air.wait_all async 
      %async_token, %results = air.execute [%4] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %async_token_1, %results_2 = air.execute [%async_token] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %5 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.get async [%arg5]  @channel_7[] (%results_2[%arg4, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 0 : i32} : (memref<128x1024xbf16, 1>)
        %9 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %8) -> (!air.async.token) {
          %15 = air.channel.put async [%arg7]  @channel_8[] (%results_2[%arg4, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.yield %15 : !air.async.token
        } {unrolled_iteration = 0 : i32}
        %10 = air.wait_all async [%9]  {async_back = true, unrolled_iteration = 0 : i32}
        %11 = arith.addi %arg4, %c256 : index
        %12 = air.channel.get async [%10]  @channel_7[] (%results[%11, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 1 : i32} : (memref<128x1024xbf16, 1>)
        %13 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %12) -> (!air.async.token) {
          %15 = air.channel.put async [%arg7]  @channel_8[] (%results[%11, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.yield %15 : !air.async.token
        } {unrolled_iteration = 1 : i32}
        %14 = air.wait_all async [%13]  {async_back = true, unrolled_iteration = 1 : i32}
        scf.yield %14 : !air.async.token
      } {unroll = 2 : i64}
      %async_token_3 = air.execute [%5] {
        memref.dealloc %results_2 : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %async_token_4 = air.execute [%5] {
        memref.dealloc %results : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %6 = air.wait_all async 
      %7 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %6) -> (!air.async.token) {
        %async_token_5, %results_6 = air.execute [%arg5] -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %8 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg7] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %9 = air.channel.get async [%async_token_8]  @channel_8[] (%results_9[] [] []) {id = 38 : i32} : (memref<128x128xbf16, 1>)
          %async_token_10 = air.execute [%9] {
            memref.dealloc %results_9 : memref<128x128xbf16, 1>
          }
          %10 = air.wait_all async [%async_token_10] 
          scf.yield %10 : !air.async.token
        }
        %async_token_7 = air.execute [%8] {
          memref.dealloc %results_6 : memref<128x128xbf16, 1>
        }
        scf.yield %async_token_7 : !air.async.token
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Check with the presence of scf.parallel loop as consumer thread.
// CHECK-LABEL: scf_parallel
// CHECK: air.segment async
// CHECK: air.segment async
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_9
// CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT3]], %[[EVENT5]]]
// CHECK: %[[EVENT7:.*]] = scf.parallel
// CHECK: scf.yield
// CHECK: %[[EVENT8:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_9
// CHECK-NEXT: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT8]], %[[EVENT9]]]
// CHECK-NEXT: %[[EVENT11:.*]] = scf.parallel
// CHECK: scf.yield
// CHECK: %[[EVENT12:.*]] = air.wait_all async [%[[EVENT11]]]
// CHECK: scf.yield %[[EVENT8]], %[[EVENT12]], %[[EVENT12]], %[[EVENT9]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token
// CHECK: air.segment async
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_11
// CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT3]], %[[EVENT5]]]
// CHECK: %[[EVENT7:.*]] = scf.parallel
// CHECK: scf.yield
// CHECK: %[[EVENT8:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_11
// CHECK-NEXT: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT8]], %[[EVENT9]]]
// CHECK-NEXT: %[[EVENT11:.*]] = scf.parallel
// CHECK: scf.yield
// CHECK: %[[EVENT12:.*]] = air.wait_all async [%[[EVENT11]]]
// CHECK: scf.yield %[[EVENT8]], %[[EVENT12]], %[[EVENT12]], %[[EVENT9]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token

air.channel @channel_12 [1, 1]
air.channel @channel_11 [1, 1]
air.channel @channel_10 [1, 1]
air.channel @channel_9 [1, 1]
func.func @scf_parallel() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
    %1 = air.segment async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %4 = air.wait_all async 
      %5 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %4) -> (!air.async.token) {
        %async_token, %results = air.execute [%arg5] -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %6 = air.channel.put async [%async_token]  @channel_9[] (%results[] [] []) {id = 16 : i32} : (memref<128x128xbf16, 1>)
        %async_token_0 = air.execute [%6] {
          memref.dealloc %results : memref<128x128xbf16, 1>
        }
        scf.yield %async_token_0 : !air.async.token
      }
      air.segment_terminator
    }
    %2 = air.segment async  attributes {id = 4 : i32, x_loc = 4 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
      %c512 = arith.constant 512 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %4 = air.wait_all async 
      %async_token, %results = air.execute [%4] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %async_token_1, %results_2 = air.execute [%async_token] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %5 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.get async [%arg5]  @channel_9[] (%results_2[%arg4, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 0 : i32} : (memref<128x1024xbf16, 1>)
        %9 = scf.parallel (%arg6) = (%c0) to (%c256) step (%c128) init (%8) -> !air.async.token {
          %15 = air.channel.put async [%8]  @channel_10[] (%results_2[%c0, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.reduce(%15)  : !air.async.token {
          ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
            %18 = air.wait_all async [%arg14, %arg15] 
            scf.reduce.return %18 : !air.async.token
          }
          scf.yield
        } {unrolled_iteration = 0 : i32}
        %10 = air.wait_all async [%9]  {async_back = true, unrolled_iteration = 0 : i32}
        %11 = arith.addi %arg4, %c256 : index
        %12 = air.channel.get async [%10]  @channel_9[] (%results[%11, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 1 : i32} : (memref<128x1024xbf16, 1>)
        %13 = scf.parallel (%arg6) = (%c0) to (%c256) step (%c128) init (%12) -> !air.async.token {
          %15 = air.channel.put async [%12]  @channel_10[] (%results[%c0, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.reduce(%15)  : !air.async.token {
          ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
            %18 = air.wait_all async [%arg14, %arg15] 
            scf.reduce.return %18 : !air.async.token
          }
          scf.yield
        } {unrolled_iteration = 1 : i32}
        %14 = air.wait_all async [%13]  {async_back = true, unrolled_iteration = 1 : i32}
        scf.yield %14 : !air.async.token
      } {unroll = 2 : i64}
      %async_token_3 = air.execute [%5] {
        memref.dealloc %results_2 : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %async_token_4 = air.execute [%5] {
        memref.dealloc %results : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %6 = air.wait_all async 
      %7 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %6) -> (!air.async.token) {
        %async_token_5, %results_6 = air.execute [%arg5] -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %8 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg7] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %10 = air.channel.get async [%async_token_8]  @channel_10[] (%results_9[] [] []) {id = 22 : i32} : (memref<128x128xbf16, 1>)
          %async_token_10 = air.execute [%10] {
            memref.dealloc %results_9 : memref<128x128xbf16, 1>
          }
          %11 = air.wait_all async [%async_token_10] 
          scf.yield %11 : !air.async.token
        }
        %9 = air.channel.put async [%8]  @channel_11[] (%results_6[] [] []) {id = 32 : i32} : (memref<128x128xbf16, 1>)
        %async_token_7 = air.execute [%9] {
          memref.dealloc %results_6 : memref<128x128xbf16, 1>
        }
        scf.yield %async_token_7 : !air.async.token
      }
      air.segment_terminator
    }
    %3 = air.segment async  attributes {id = 6 : i32, x_loc = 8 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
      %c512 = arith.constant 512 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %4 = air.wait_all async 
      %async_token, %results = air.execute [%4] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %async_token_1, %results_2 = air.execute [%async_token] -> (memref<128x1024xbf16, 1>) {
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        air.execute_terminator %alloc : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %5 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.get async [%arg5]  @channel_11[] (%results_2[%arg4, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 0 : i32} : (memref<128x1024xbf16, 1>)
        %9 = scf.parallel (%arg6) = (%c0) to (%c256) step (%c128) init (%8) -> !air.async.token {
          %15 = air.channel.put async [%8]  @channel_12[] (%results_2[%arg4, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.reduce(%15)  : !air.async.token {
          ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
            %18 = air.wait_all async [%arg14, %arg15] 
            scf.reduce.return %18 : !air.async.token
          }
          scf.yield
        } {unrolled_iteration = 0 : i32}
        %10 = air.wait_all async [%9]  {async_back = true, unrolled_iteration = 0 : i32}
        %11 = arith.addi %arg4, %c256 : index
        %12 = air.channel.get async [%10]  @channel_11[] (%results[%11, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, unrolled_iteration = 1 : i32} : (memref<128x1024xbf16, 1>)
        %13 = scf.parallel (%arg6) = (%c0) to (%c256) step (%c128) init (%12) -> !air.async.token {
          %15 = air.channel.put async [%12]  @channel_12[] (%results[%11, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
          scf.reduce(%15)  : !air.async.token {
          ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
            %18 = air.wait_all async [%arg14, %arg15] 
            scf.reduce.return %18 : !air.async.token
          }
          scf.yield
        } {unrolled_iteration = 1 : i32}
        %14 = air.wait_all async [%13]  {async_back = true, unrolled_iteration = 1 : i32}
        scf.yield %14 : !air.async.token
      } {unroll = 2 : i64}
      %async_token_3 = air.execute [%5] {
        memref.dealloc %results_2 : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 0 : i32}
      %async_token_4 = air.execute [%5] {
        memref.dealloc %results : memref<128x1024xbf16, 1>
      } {unrolled_iteration = 1 : i32}
      %6 = air.wait_all async 
      %7 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %6) -> (!air.async.token) {
        %async_token_5, %results_6 = air.execute [%arg5] -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %8 = scf.for %arg6 = %c0 to %c256 step %c128 iter_args(%arg7 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg7] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %9 = air.channel.get async [%async_token_8]  @channel_12[] (%results_9[] [] []) {id = 38 : i32} : (memref<128x128xbf16, 1>)
          %async_token_10 = air.execute [%9] {
            memref.dealloc %results_9 : memref<128x128xbf16, 1>
          }
          %10 = air.wait_all async [%async_token_10] 
          scf.yield %10 : !air.async.token
        }
        %async_token_7 = air.execute [%8] {
          memref.dealloc %results_6 : memref<128x128xbf16, 1>
        }
        scf.yield %async_token_7 : !air.async.token
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Check with targetting multiple scf.for loops which are under the same scope.
// CHECK-LABEL: multiple_ping_pong
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT6:.*]] = {{.*}} %[[EVENT7:.*]] = {{.*}} %[[EVENT8:.*]] = {{.*}} %[[EVENT9:.*]] = {{.*}})

module {
  func.func @multiple_ping_pong() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async  attributes {id = 2 : i32} {
        %c8 = arith.constant 8 : index
        %c448 = arith.constant 448 : index
        %c114688 = arith.constant 114688 : index
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c12544 = arith.constant 12544 : index
        %2 = air.wait_all async 
        %async_token, %results = air.execute [%2] -> (memref<1x256x112x4xi8, 1>) {
          %alloc = memref.alloc() : memref<1x256x112x4xi8, 1>
          air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 1 : i32}
        %async_token_1, %results_2 = air.execute [%async_token] -> (memref<1x256x112x4xi8, 1>) {
          %alloc = memref.alloc() : memref<1x256x112x4xi8, 1>
          air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 0 : i32}
        %3 = scf.for %arg4 = %c0 to %c112 step %c8 iter_args(%arg5 = %async_token_1) -> (!air.async.token) {
          %6 = air.channel.get async [%arg5]  @channel_0[] (%results_2[] [] []) {async_front = true, id = 2 : i32, unrolled_iteration = 0 : i32} : (memref<1x256x112x4xi8, 1>)
          %7 = scf.for %arg6 = %c0 to %c4 step %c1_0 iter_args(%arg7 = %6) -> (!air.async.token) {
            %12 = air.channel.put async [%arg7]  @channel_1[%c0, %c0] (%results_2[%c0, %c0, %c0, %arg6] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 3 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %12 : !air.async.token
          } {unrolled_iteration = 0 : i32}
          %8 = air.wait_all async [%7]  {async_back = true, unrolled_iteration = 0 : i32}
          %9 = air.channel.get async [%8]  @channel_0[] (%results[] [] []) {async_front = true, id = 2 : i32, unrolled_iteration = 1 : i32} : (memref<1x256x112x4xi8, 1>)
          %10 = scf.for %arg6 = %c0 to %c4 step %c1_0 iter_args(%arg7 = %9) -> (!air.async.token) {
            %12 = air.channel.put async [%arg7]  @channel_1[%c0, %c0] (%results[%c0, %c0, %c0, %arg6] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 3 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %12 : !air.async.token
          } {unrolled_iteration = 1 : i32}
          %11 = air.wait_all async [%10]  {async_back = true, unrolled_iteration = 1 : i32}
          scf.yield %11 : !air.async.token
        } {isolated = true, unroll = 2 : i32}
        %async_token_3 = air.execute [%3] {
          memref.dealloc %results_2 : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 0 : i32}
        %async_token_4 = air.execute [%3] {
          memref.dealloc %results : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 1 : i32}
        %4 = air.wait_all async 
        %async_token_5, %results_6 = air.execute [%4] -> (memref<1x256x112x4xi8, 1>) {
          %alloc = memref.alloc() : memref<1x256x112x4xi8, 1>
          air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 1 : i32}
        %async_token_7, %results_8 = air.execute [%async_token_5] -> (memref<1x256x112x4xi8, 1>) {
          %alloc = memref.alloc() : memref<1x256x112x4xi8, 1>
          air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 0 : i32}
        %5 = scf.for %arg4 = %c0 to %c112 step %c8 iter_args(%arg5 = %async_token_7) -> (!air.async.token) {
          %6 = air.channel.get async [%arg5]  @channel_8[] (%results_8[] [] []) {async_front = true, id = 2 : i32, unrolled_iteration = 0 : i32} : (memref<1x256x112x4xi8, 1>)
          %7 = scf.for %arg6 = %c0 to %c4 step %c1_0 iter_args(%arg7 = %6) -> (!air.async.token) {
            %12 = air.channel.put async [%arg7]  @channel_6[%c0, %c0] (%results_8[%c0, %c0, %c0, %arg6] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 4 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %12 : !air.async.token
          } {unrolled_iteration = 0 : i32}
          %8 = air.wait_all async [%7]  {async_back = true, unrolled_iteration = 0 : i32}
          %9 = air.channel.get async [%8]  @channel_8[] (%results_6[] [] []) {async_front = true, id = 2 : i32, unrolled_iteration = 1 : i32} : (memref<1x256x112x4xi8, 1>)
          %10 = scf.for %arg6 = %c0 to %c4 step %c1_0 iter_args(%arg7 = %9) -> (!air.async.token) {
            %12 = air.channel.put async [%arg7]  @channel_6[%c0, %c0] (%results_6[%c0, %c0, %c0, %arg6] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 4 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %12 : !air.async.token
          } {unrolled_iteration = 1 : i32}
          %11 = air.wait_all async [%10]  {async_back = true, unrolled_iteration = 1 : i32}
          scf.yield %11 : !air.async.token
        } {isolated = true, unroll = 2 : i32}
        %async_token_9 = air.execute [%5] {
          memref.dealloc %results_8 : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 0 : i32}
        %async_token_10 = air.execute [%5] {
          memref.dealloc %results_6 : memref<1x256x112x4xi8, 1>
        } {unrolled_iteration = 1 : i32}
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
