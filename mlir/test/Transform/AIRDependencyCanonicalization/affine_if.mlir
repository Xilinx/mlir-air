//===- affine_if.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// CHECK: %[[EVENT0:.*]] = air.wait_all async
// CHECK-NEXT: %[[EVENT1:.*]] = affine.if
// CHECK-NEXT: %[[EVENT2:.*]] = air.channel.put async [%[[EVENT0]]]
// CHECK-NEXT: %[[EVENT3:.*]] = air.wait_all async [%[[EVENT2]]]
// CHECK-NEXT: affine.yield %[[EVENT3]]
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[EVENT4:.*]] = air.wait_all async [%[[EVENT0]]]
// CHECK-NEXT: %[[EVENT5:.*]] = affine.if
// CHECK: %[[EVENT6:.*]] = air.channel.get async [%[[EVENT4]]]
// CHECK-NEXT: %[[EVENT7:.*]] = air.channel.put async [%[[EVENT6]]]
// CHECK-NEXT: %[[EVENT8:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK-NEXT: affine.yield %[[EVENT8]]
// CHECK-NEXT: } else {
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT4]]]
// CHECK-NEXT: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT9]]]
// CHECK-NEXT: affine.yield %[[EVENT10]]

#set = affine_set<()[s0] : (s0 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
module {
  air.channel @channel_0 [4]
  func.func @affine_if_1(%arg0: memref<1x1x2048xi32>, %arg1: memref<1x1x2048xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = air.herd @herd_0 async  tile (%arg2, %arg3) in (%arg4=%c4, %arg5=%c1) attributes {id = 1 : i32} {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %async_token, %results = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
      } {id = 1 : i32}
      %1 = air.wait_all async [%async_token, %async_token]  {id = 4 : i32}
      %2 = affine.if #set()[%arg2] -> !air.async.token {
        %3 = air.channel.put async [%1, %async_token]  @channel_0[%arg2] (%results[] [] []) {id = 1 : i32} : (memref<1x1x2048xi32, 2 : i32>)
        %4 = air.wait_all async [%3, %async_token, %1]  {id = 5 : i32}
        affine.yield %4 : !air.async.token
      } else {
        %3 = air.wait_all async [%async_token, %1, %1]  {id = 1 : i32}
        %4 = affine.if #set1()[%arg2] -> !air.async.token {
          %c1_0 = arith.constant 1 : index
          %6 = arith.subi %arg2, %c1_0 : index
          %7 = air.channel.get async [%3, %async_token, %1]  @channel_0[%6] (%results[] [] []) {id = 2 : i32} : (memref<1x1x2048xi32, 2 : i32>)
          %8 = air.channel.put async [%7, %3, %async_token, %1]  @channel_0[%arg2] (%results[] [] []) {id = 3 : i32} : (memref<1x1x2048xi32, 2 : i32>)
          %9 = air.wait_all async [%8, %7, %3, %async_token, %1]  {id = 2 : i32}
          affine.yield %9 : !air.async.token
        } else {
          %c1_0 = arith.constant 1 : index
          %6 = arith.subi %arg2, %c1_0 : index
          %7 = air.channel.get async [%3, %async_token, %1]  @channel_0[%6] (%results[] [] []) {id = 4 : i32} : (memref<1x1x2048xi32, 2 : i32>)
          %8 = air.wait_all async [%7, %3, %async_token, %1]  {id = 3 : i32}
          affine.yield %8 : !air.async.token
        }
        %5 = air.wait_all async [%3, %async_token, %1]  {id = 6 : i32}
        affine.yield %5 : !air.async.token
      }
    }
    return
  }
}

