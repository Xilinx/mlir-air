//===- segment_loop_fusion.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-loop-fusion %s | FileCheck %s

// Fuse scf.for loops in air.segment.

// CHECK-LABEL: func.func @func0
// CHECK: air.launch
// CHECK: air.segment
// CHECK: memref.alloc()
// CHECK: scf.for
// CHECK: memref.alloc()
// CHECK: memref.alloc()
// CHECK: %[[EVENT0:.*]] = air.channel.get{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT1:.*]] = air.channel.put{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT2:.*]] = air.channel.put{{.*}}@channel_
// CHECK: %[[EVENT3:.*]] = air.channel.get{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT3]])
// CHECK: %[[EVENT4:.*]] = air.channel.put{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT3]])
// CHECK: %[[EVENT5:.*]] = air.channel.put{{.*}}@channel_

func.func @func0() {
  %c32 = arith.constant 32 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) {
    %6 = air.segment @segment_0 async {
      %c32_9 = arith.constant 32 : index
      %c256_10 = arith.constant 256 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c2048_13 = arith.constant 2048 : index
      %c64_14 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %7 = air.wait_all async 
      %8 = air.wait_all async 
      %async_token_15, %results_16 = air.execute -> (memref<64x64xi32, 1>) {
        %alloc = memref.alloc() : memref<64x64xi32, 1>
        air.execute_terminator %alloc : memref<64x64xi32, 1>
      }
      %async_token_17, %results_18 = air.execute [%async_token_15] -> (memref<64x256xi32, 1>) {
        %alloc = memref.alloc() : memref<64x256xi32, 1>
        air.execute_terminator %alloc : memref<64x256xi32, 1>
      }
      %async_token_19, %results_20 = air.execute [%async_token_17] -> (memref<256x64xi32, 1>) {
        %alloc = memref.alloc() : memref<256x64xi32, 1>
        air.execute_terminator %alloc : memref<256x64xi32, 1>
      }
      %9 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_0[] (%results_18[%c0_11, %arg12] [%c32_9, %c32_9] [%c256_10, %c1_12]) : (memref<64x256xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %10 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_1[] (%results_18[%c32_9, %arg12] [%c32_9, %c32_9] [%c256_10, %c1_12]) : (memref<64x256xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %11 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_19) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_2[] (%results_20[%arg12, %c0_11] [%c32_9, %c32_9] [%c64_14, %c1_12]) : (memref<256x64xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %12 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_19) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_3[] (%results_20[%arg12, %c32_9] [%c32_9, %c32_9] [%c64_14, %c1_12]) : (memref<256x64xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %13 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = air.channel.get async [%arg11, %7]  @channel_4[] (%results_18[] [] []) : (memref<64x256xi32, 1>)
        scf.yield %18 : !air.async.token
      }
      %14 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_19) -> (!air.async.token) {
        %18 = air.channel.get async [%arg11, %8]  @channel_5[] (%results_20[] [] []) : (memref<256x64xi32, 1>)
        scf.yield %18 : !air.async.token
      }
      %async_token_21 = air.execute [%async_token_19] {
        memref.dealloc %results_20 : memref<256x64xi32, 1>
      }
      %async_token_22 = air.execute [%async_token_17] {
        memref.dealloc %results_18 : memref<64x256xi32, 1>
      }
      %async_token_23 = air.execute [%7, %8] {
        memref.dealloc %results_16 : memref<64x64xi32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Memref shrinkage via data access pattern analysis.

// CHECK-LABEL: func.func @func1
// CHECK: air.launch
// CHECK: air.segment
// CHECK: scf.for
// CHECK: memref.alloc() : memref<64x256xi32, 1>
// CHECK: %[[EVENT0:.*]] = air.channel.get {{.*}} : (memref<64x256xi32, 1>)
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT1:.*]] = air.channel.put {{.*}} : (memref<64x256xi32, 1>)
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT2:.*]] = air.channel.put {{.*}} : (memref<64x256xi32, 1>)

func.func @func1() {
  %c32 = arith.constant 32 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) {
    %6 = air.segment @segment_0 async {
      %c32_9 = arith.constant 32 : index
      %c256_10 = arith.constant 256 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c2048_13 = arith.constant 2048 : index
      %c64_14 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %7 = air.wait_all async 
      %8 = air.wait_all async 
      %async_token_17, %results_18 = air.execute -> (memref<64x2048xi32, 1>) {
        %alloc = memref.alloc() : memref<64x2048xi32, 1>
        air.execute_terminator %alloc : memref<64x2048xi32, 1>
      }
      %9 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_0[] (%results_18[%c0_11, %arg12] [%c32_9, %c32_9] [%c2048_13, %c1_12]) : (memref<64x2048xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %10 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_1[] (%results_18[%c32_9, %arg12] [%c32_9, %c32_9] [%c2048_13, %c1_12]) : (memref<64x2048xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %13 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = air.channel.get async [%arg11, %7]  @channel_4[] (%results_18[%c0_11, %c0_11] [%c64_14, %c256_10] [%c2048_13, %c1_12]) : (memref<64x2048xi32, 1>)
        scf.yield %18 : !air.async.token
      }
      %async_token_22 = air.execute [%async_token_17] {
        memref.dealloc %results_18 : memref<64x2048xi32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
