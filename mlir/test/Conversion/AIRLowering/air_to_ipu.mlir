//===- air_to_ipu.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std -cse --split-input-file | FileCheck %s

// CHECK-LABEL: aie.device(ipu)
// CHECK: {sym_name = "segment0"}
// CHECK: func.func @func0(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>)
// CHECK: %[[CST_0:.*]] = arith.constant 0 : i64
// CHECK: %[[CST_1:.*]] = arith.constant 1 : i64
// CHECK: %[[CST_2:.*]] = arith.constant 2 : i32
// CHECK: %[[CST_64:.*]] = arith.constant 64 : i64
// CHECK: airrt.dma_memcpy_nd(%[[CST_2]], %[[CST_0]], %[[CST_0]], %[[VAL_0]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]], [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]], [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: %[[VAL_2:.*]] = airrt.segment_load "segment0" : i64
// CHECK: %[[CST_7:.*]] = arith.constant 7 : i32
// CHECK: airrt.dma_memcpy_nd(%[[CST_7]], %[[CST_0]], %[[CST_0]], %[[VAL_1]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]], [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]], [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])

module {
  aie.device(ipu) {
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  } {sym_name = "segment0"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    air.channel.put  @channel_0[] (%arg0[] [] []) {id = 1 : i32, metadata = @airMemcpyId2} : (memref<64xi32>)
    air.segment @segment0  {
      %c1 = arith.constant 1 : index
      %alloc = memref.alloc() : memref<64xi32, 1>
      air.channel.get  @channel_0[] (%alloc[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
      air.channel.put  @channel_1[] (%alloc[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      memref.dealloc %alloc : memref<64xi32, 1>
      air.herd @func4  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) attributes {x_loc = 0 : i32, y_loc = 2 : i32} {
        %c1_i32 = arith.constant 1 : i32
        %alloc_1 = memref.alloc() : memref<64xi32, 2>
        %alloc_2 = memref.alloc() : memref<64xi32, 2>
        air.channel.get  @channel_1[%arg2, %arg3] (%alloc_1[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        affine.for %arg6 = 0 to 64 {
          %0 = affine.load %alloc_1[%arg6] : memref<64xi32, 2>
          %1 = arith.addi %0, %c1_i32 : i32
          affine.store %1, %alloc_2[%arg6] : memref<64xi32, 2>
        }
        air.channel.put  @channel_2[%arg2, %arg3] (%alloc_2[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
        memref.dealloc %alloc_1 : memref<64xi32, 2>
        memref.dealloc %alloc_2 : memref<64xi32, 2>
        air.herd_terminator
      }
      %alloc_0 = memref.alloc() : memref<64xi32, 1>
      air.channel.get  @channel_2[] (%alloc_0[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
      air.channel.put  @channel_3[] (%alloc_0[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
      memref.dealloc %alloc_0 : memref<64xi32, 1>
      air.segment_terminator
    }
    air.channel.get  @channel_3[] (%arg1[] [] []) {id = 8 : i32, metadata = @airMemcpyId7} : (memref<64xi32>)
    return
  }
}

// -----

// Asynchronous version

// CHECK-LABEL: aie.device(ipu)
// CHECK: {sym_name = "segment0"}
// CHECK: func.func @func0(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>)
// CHECK: %[[CST_0:.*]] = arith.constant 0 : i64
// CHECK: %[[CST_1:.*]] = arith.constant 1 : i64
// CHECK: %[[CST_2:.*]] = arith.constant 2 : i32
// CHECK: %[[CST_64:.*]] = arith.constant 64 : i64
// CHECK: airrt.dma_memcpy_nd(%[[CST_2]], %[[CST_0]], %[[CST_0]], %[[VAL_0]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]], [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]], [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: %[[VAL_2:.*]] = airrt.segment_load "segment0" : i64
// CHECK: %[[CST_7:.*]] = arith.constant 7 : i32
// CHECK: airrt.dma_memcpy_nd(%[[CST_7]], %[[CST_0]], %[[CST_0]], %[[VAL_1]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]], [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]], [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])

module {
  aie.device(ipu) {
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  } {sym_name = "segment0"}
  airrt.module_metadata{
  }
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %0 = air.channel.put async  @channel_0[] (%arg0[] [] []) {id = 1 : i32, metadata = @airMemcpyId2} : (memref<64xi32>)
    %1 = air.segment @segment0 async  attributes {id = 2 : i32} {
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 1 : i32}
      %3 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
      %4 = air.channel.put async [%3]  @channel_1[] (%results[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      %async_token_0 = air.execute [%4] {
        memref.dealloc %results : memref<64xi32, 1>
      } {id = 2 : i32}
      %5 = air.herd @func4 async  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) attributes {id = 1 : i32, x_loc = 0 : i32, y_loc = 2 : i32} {
        %c1_i32 = arith.constant 1 : i32
        %async_token_4, %results_5 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 3 : i32}
        %async_token_6, %results_7 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 4 : i32}
        %8 = air.channel.get async [%async_token_4]  @channel_1[%arg2, %arg3] (%results_5[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        %async_token_8 = air.execute [%async_token_6, %8] {
          affine.for %arg6 = 0 to 64 {
            %10 = affine.load %results_5[%arg6] : memref<64xi32, 2>
            %11 = arith.addi %10, %c1_i32 : i32
            affine.store %11, %results_7[%arg6] : memref<64xi32, 2>
          }
        } {id = 5 : i32}
        %9 = air.channel.put async [%async_token_8]  @channel_2[%arg2, %arg3] (%results_7[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
        %async_token_9 = air.execute [%async_token_8] {
          memref.dealloc %results_5 : memref<64xi32, 2>
        } {id = 6 : i32}
        %async_token_10 = air.execute [%9] {
          memref.dealloc %results_7 : memref<64xi32, 2>
        } {id = 7 : i32}
        air.herd_terminator
      }
      %async_token_1, %results_2 = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 8 : i32}
      %6 = air.channel.get async [%async_token_1]  @channel_2[] (%results_2[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
      %7 = air.channel.put async [%6]  @channel_3[] (%results_2[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
      %async_token_3 = air.execute [%7] {
        memref.dealloc %results_2 : memref<64xi32, 1>
      } {id = 9 : i32}
      air.segment_terminator
    }
    %2 = air.channel.get async  @channel_3[] (%arg1[] [] []) {id = 8 : i32, metadata = @airMemcpyId7} : (memref<64xi32>)
    return
  }
}
