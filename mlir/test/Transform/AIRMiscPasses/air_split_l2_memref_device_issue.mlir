//===- air_split_l2_memref_device_issue.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-l2-memref="tiles-per-l2-tile=1" | FileCheck %s

// A core starts a put from any row of an L2 buffer, which also sends a fixed
// row elsewhere. The core's offset spans the whole buffer, so it stays whole.

// CHECK: memref.alloc() : memref<8x256xi32, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<{{[0-9]+}}x256xi32, 1 : i32>
// CHECK: air.channel.put async {{.*}}@Fixed[] (%{{.*}}[5, 0] [1, 256] [256, 1])
// CHECK: air.channel.put async {{.*}}@Row[] (%{{.*}}[0, 0] [1, 256] [256, 1]) {air.device_issue
module {
  air.channel @WIn []
  air.channel @IdxIn []
  air.channel @Row []
  air.channel @Fixed []
  air.channel @YOut []
  air.channel @YOut2 []
  func.func @split(%arg0: memref<8x256xi32>, %arg1: memref<16xi32>, %arg2: memref<256xi32>, %arg3: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<8x256xi32>, memref<16xi32>, memref<256xi32>, memref<256xi32> attributes {id = 1 : i32} {
      %1 = air.channel.put async  @WIn[] (%arg8[] [] []) {id = 1 : i32} : (memref<8x256xi32>)
      %2 = air.channel.put async  @IdxIn[] (%arg9[] [] []) {id = 2 : i32} : (memref<16xi32>)
      %3 = air.channel.get async  @YOut[] (%arg10[] [] []) {id = 3 : i32} : (memref<256xi32>)
      %4 = air.channel.get async  @YOut2[] (%arg11[] [] []) {id = 4 : i32} : (memref<256xi32>)
      %5 = air.segment @seg async  attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<8x256xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x256xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x256xi32, 1 : i32>
        }
        %6 = air.channel.get async [%async_token]  @WIn[] (%results[] [] []) {id = 5 : i32} : (memref<8x256xi32, 1 : i32>)
        %7 = air.channel.put async [%6]  @Fixed[] (%results[5, 0] [1, 256] [256, 1]) {id = 6 : i32} : (memref<8x256xi32, 1 : i32>)
        %8 = air.channel.put async [%7]  @Row[] (%results[0, 0] [1, 256] [256, 1]) {air.device_issue = 2 : i64, id = 7 : i32} : (memref<8x256xi32, 1 : i32>)
        %9 = air.herd @fwd async [%7]  tile (%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c1_0) attributes {id = 3 : i32} {
          %c256_i64 = arith.constant 256 : i64
          %c0 = arith.constant 0 : index
          %async_token_2, %results_3 = air.execute -> (memref<16xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xi32, 2 : i32>
            air.execute_terminator %alloc : memref<16xi32, 2 : i32>
          }
          %async_token_4, %results_5 = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %11 = air.channel.get async [%async_token_2]  @IdxIn[] (%results_3[] [] []) {id = 8 : i32} : (memref<16xi32, 2 : i32>)
          %async_token_6, %results_7 = air.execute [%11] -> (i32) {
            %17 = memref.load %results_3[%c0] : memref<16xi32, 2 : i32>
            air.execute_terminator %17 : i32
          }
          %async_token_8 = air.execute [%async_token_6] {
            memref.dealloc %results_3 : memref<16xi32, 2 : i32>
          }
          %12 = arith.index_cast %results_7 : i32 to index
          %13 = arith.index_cast %12 : index to i64
          %14 = arith.muli %13, %c256_i64 : i64
          %async_token_9 = air.execute {
            air.channel.issue  @Row[] offset(%14 : i64) {issue_id = 2 : i64}
          }
          %15 = air.channel.get async [%async_token_4]  @Row[] (%results_5[] [] []) {id = 9 : i32} : (memref<256xi32, 2 : i32>)
          %16 = air.channel.put async [%15]  @YOut[] (%results_5[] [] []) {id = 10 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_10 = air.execute [%16] {
            memref.dealloc %results_5 : memref<256xi32, 2 : i32>
          }
        }
        %async_token_1 = air.execute [%8] {
          memref.dealloc %results : memref<8x256xi32, 1 : i32>
        }
        %10 = air.herd @fwd2 async  tile (%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c1_0) attributes {id = 4 : i32} {
          %async_token_2, %results_3 = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %11 = air.channel.get async [%async_token_2]  @Fixed[] (%results_3[] [] []) {id = 11 : i32} : (memref<256xi32, 2 : i32>)
          %12 = air.channel.put async [%11]  @YOut2[] (%results_3[] [] []) {id = 12 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_4 = air.execute [%12] {
            memref.dealloc %results_3 : memref<256xi32, 2 : i32>
          }
        }
      }
    }
    return
  }
}
