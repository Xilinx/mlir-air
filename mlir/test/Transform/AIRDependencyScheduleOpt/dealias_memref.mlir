//===- dealias_memref.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-dealias-memref %s | FileCheck %s

// Duplication of intermediate memref to reduce the number of memory access per
// memref (buffer).

// CHECK-LABEL:   @func0
// CHECK:   air.segment @segment0
// CHECK:   %[[ASYNC_TOKEN_0:.*]], %[[VAL_0:.*]] = air.execute -> (memref<64xi32, 1>) {
// CHECK:   memref.alloc() : memref<64xi32, 1>
// CHECK:   %[[ASYNC_TOKEN_1:.*]], %[[VAL_1:.*]] = air.execute -> (memref<64xi32, 1>) {
// CHECK:   memref.alloc() : memref<64xi32, 1>
// CHECK:   air.channel.get async{{.*}}@channel_0[] (%[[VAL_0]][] [] []) {{.*}} (memref<64xi32, 1>)
// CHECK:   air.channel.put async{{.*}}@channel_1[] (%[[VAL_0]][] [] []) {{.*}} (memref<64xi32, 1>)
// CHECK:   air.channel.get async{{.*}}@channel_2[] (%[[VAL_1]][] [] []) {{.*}} (memref<64xi32, 1>)
// CHECK:   air.channel.put async{{.*}}@channel_3[] (%[[VAL_1]][] [] []) {{.*}} (memref<64xi32, 1>)
// CHECK:   %[[ASYNC_TOKEN_2:.*]] = air.execute
// CHECK:     memref.dealloc %[[VAL_0]] : memref<64xi32, 1>
// CHECK:   }
// CHECK:   %[[ASYNC_TOKEN_3:.*]] = air.execute
// CHECK:     memref.dealloc %[[VAL_1]] : memref<64xi32, 1>
// CHECK:   } {id = 2 : i32}

#map = affine_map<(d0) -> (d0)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %0 = air.channel.put async  @channel_0[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
    %1 = air.segment @segment0 async  attributes {id = 2 : i32} {
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (memref<64xi32, 1>) {
        %alloc = memref.alloc() : memref<64xi32, 1>
        air.execute_terminator %alloc : memref<64xi32, 1>
      } {id = 1 : i32}
      %3 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
      %4 = air.channel.put async [%3]  @channel_1[] (%results[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
      %5 = air.herd @func4 async  tile (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_1) attributes {id = 1 : i32} {
        %async_token_6, %results_7 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 3 : i32}
        %async_token_8, %results_9 = air.execute -> (memref<64xi32, 2>) {
          %alloc = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %alloc : memref<64xi32, 2>
        } {id = 4 : i32}
        %8 = air.channel.get async [%async_token_6]  @channel_1[%arg2, %arg3] (%results_7[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        %async_token_10 = air.execute [%async_token_8, %8] {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%results_7 : memref<64xi32, 2>) outs(%results_9 : memref<64xi32, 2>) {
          ^bb0(%in: i32, %out: i32):
            %c1_i32 = arith.constant 1 : i32
            %10 = arith.addi %in, %c1_i32 : i32
            linalg.yield %10 : i32
          }
        } {id = 5 : i32}
        %9 = air.channel.put async [%async_token_10]  @channel_2[%arg2, %arg3] (%results_9[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
        %async_token_11 = air.execute [%async_token_10] {
          memref.dealloc %results_7 : memref<64xi32, 2>
        } {id = 6 : i32}
        %async_token_12 = air.execute [%9] {
          memref.dealloc %results_9 : memref<64xi32, 2>
        } {id = 7 : i32}
      }
      %6 = air.channel.get async  @channel_2[] (%results[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
      %7 = air.channel.put async [%6]  @channel_3[] (%results[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
      %async_token_2 = air.execute [%7] {
        memref.dealloc %results : memref<64xi32, 1>
      } {id = 2 : i32}
    }
    %2 = air.channel.get async  @channel_3[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
    return
  }
}
