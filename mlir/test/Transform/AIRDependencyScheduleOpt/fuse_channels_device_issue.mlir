//===- fuse_channels_device_issue.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels | FileCheck %s

// A core starts each of two puts whose static parts are the same. Each channel
// keeps its own descriptor for the core to patch, so the two are not fused.

// CHECK-DAG: air.channel.put async {{.*}}@WA[]
// CHECK-DAG: air.channel.put async {{.*}}@WB[]
// CHECK-DAG: air.channel.get async {{.*}}@WA[]
// CHECK-DAG: air.channel.get async {{.*}}@WB[]
module {
  air.channel @IdxIn [] {channel_type = "npu_dma_packet"}
  air.channel @WA [] {channel_type = "npu_dma_packet"}
  air.channel @WB [] {channel_type = "npu_dma_packet"}
  air.channel @YOut []
  func.func @fuse(%arg0: memref<8x256xi32>, %arg1: memref<16xi32>, %arg2: memref<2x256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<8x256xi32>, memref<16xi32>, memref<2x256xi32> attributes {id = 1 : i32} {
      %1 = air.channel.put async  @IdxIn[] (%arg8[] [] []) {id = 1 : i32} : (memref<16xi32>)
      %2 = air.channel.put async  @WA[] (%arg7[0, 0] [1, 256] [256, 1]) {air.device_issue = 1 : i64, id = 2 : i32} : (memref<8x256xi32>)
      %3 = air.channel.put async  @WB[] (%arg7[0, 0] [1, 256] [256, 1]) {air.device_issue = 2 : i64, id = 3 : i32} : (memref<8x256xi32>)
      %4 = air.channel.get async  @YOut[] (%arg9[] [] []) {id = 4 : i32} : (memref<2x256xi32>)
      %5 = air.segment @seg async  attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %6 = air.herd @router async  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) attributes {id = 3 : i32} {
          %c256_i64 = arith.constant 256 : i64
          %c1_1 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<16xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xi32, 2 : i32>
            air.execute_terminator %alloc : memref<16xi32, 2 : i32>
          }
          %8 = air.channel.get async [%async_token]  @IdxIn[] (%results[] [] []) {id = 5 : i32} : (memref<16xi32, 2 : i32>)
          %async_token_2, %results_3 = air.execute [%8] -> (i32) {
            %15 = memref.load %results[%c0] : memref<16xi32, 2 : i32>
            air.execute_terminator %15 : i32
          }
          %9 = arith.index_cast %results_3 : i32 to index
          %10 = arith.index_cast %9 : index to i64
          %11 = arith.muli %10, %c256_i64 : i64
          %async_token_4 = air.execute {
            air.channel.issue  @WA[] offset(%11 : i64) {issue_id = 1 : i64}
          }
          %async_token_5, %results_6 = air.execute [%async_token_2] -> (i32) {
            %15 = memref.load %results[%c1_1] : memref<16xi32, 2 : i32>
            air.execute_terminator %15 : i32
          }
          %async_token_7 = air.execute [%async_token_5] {
            memref.dealloc %results : memref<16xi32, 2 : i32>
          }
          %12 = arith.index_cast %results_6 : i32 to index
          %13 = arith.index_cast %12 : index to i64
          %14 = arith.muli %13, %c256_i64 : i64
          %async_token_8 = air.execute {
            air.channel.issue  @WB[] offset(%14 : i64) {issue_id = 2 : i64}
          }
        }
        %7 = air.herd @fwd async  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) attributes {id = 4 : i32} {
          %async_token, %results = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %8 = air.channel.get async [%async_token]  @WA[] (%results[] [] []) {id = 6 : i32} : (memref<256xi32, 2 : i32>)
          %9 = air.channel.put async [%8]  @YOut[] (%results[] [] []) {id = 7 : i32} : (memref<256xi32, 2 : i32>)
          %10 = air.channel.get async [%9]  @WB[] (%results[] [] []) {id = 8 : i32} : (memref<256xi32, 2 : i32>)
          %11 = air.channel.put async [%10]  @YOut[] (%results[] [] []) {id = 9 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_1 = air.execute [%11] {
            memref.dealloc %results : memref<256xi32, 2 : i32>
          }
        }
      }
    }
    return
  }
}
