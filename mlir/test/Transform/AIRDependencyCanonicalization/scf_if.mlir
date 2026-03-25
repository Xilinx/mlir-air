//===- scf_if.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// Test that air-dependency-canonicalize handles scf.if ops returning
// !air.async.token inside a herd body.

// CHECK-LABEL: func.func @scf_if_in_herd
// CHECK: air.herd
// CHECK: %[[EVENT0:.*]] = scf.if
// CHECK: %[[EVENT1:.*]] = air.channel.get async
// CHECK: scf.yield %[[EVENT1]]
// CHECK: } else {
// CHECK: %[[EVENT2:.*]] = air.channel.get async
// CHECK: scf.yield %[[EVENT2]]
// CHECK: }
// CHECK: air.channel.put async [%[[EVENT0]]]

// CHECK-LABEL: func.func @scf_if_in_scf_for
// CHECK: air.herd
// CHECK: scf.for
// CHECK: %[[IF_RESULT:.*]] = scf.if
// CHECK: air.channel.get async
// CHECK: } else {
// CHECK: air.channel.get async
// CHECK: }
// CHECK: air.channel.put async [%[[IF_RESULT]]]

module {
  air.channel @channel_A [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @channel_B [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @channel_out [4, 1]
  func.func @scf_if_in_herd(%arg0: memref<1x1x2048xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = air.herd @herd_0 async  tile (%arg2, %arg3) in (%arg4=%c4, %arg5=%c1) args(%arg6=%arg0) : memref<1x1x2048xi32> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %async_token, %results = air.execute -> (memref<64x64xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
      } {id = 1 : i32}
      %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
      } {id = 2 : i32}
      // scf.if selecting between channels based on tile position
      %cond = arith.cmpi eq, %arg2, %c0 : index
      %1 = scf.if %cond -> (!air.async.token) {
        %2 = air.channel.get async [%async_token]  @channel_A[%c0, %c0, %arg2] (%results[] [] []) {id = 1 : i32} : (memref<64x64xbf16, 2 : i32>)
        scf.yield %2 : !air.async.token
      } else {
        %2 = air.channel.get async [%async_token]  @channel_B[%c0, %c0, %arg2] (%results[] [] []) {id = 2 : i32} : (memref<64x64xbf16, 2 : i32>)
        scf.yield %2 : !air.async.token
      }
      // Use the scf.if result as a dependency
      %3 = air.channel.put async [%1]  @channel_out[%arg2, %c0] (%results_1[] [] []) {id = 3 : i32} : (memref<64x64xbf16, 2 : i32>)
      %async_token_2 = air.execute [%3] {
        memref.dealloc %results : memref<64x64xbf16, 2 : i32>
      } {id = 3 : i32}
      %async_token_3 = air.execute [%3] {
        memref.dealloc %results_1 : memref<64x64xbf16, 2 : i32>
      } {id = 4 : i32}
    }
    return
  }
  func.func @scf_if_in_scf_for(%arg0: memref<1x1x2048xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = air.herd @herd_1 async  tile (%arg2, %arg3) in (%arg4=%c4, %arg5=%c1) args(%arg6=%arg0) : memref<1x1x2048xi32> attributes {id = 2 : i32} {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<64x64xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
      } {id = 5 : i32}
      %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
      } {id = 6 : i32}
      %cond = arith.cmpi eq, %arg2, %c0 : index
      // scf.for loop with scf.if inside selecting channels per iteration
      %1 = scf.for %iv = %c0 to %c2 step %c1_0 iter_args(%dep = %async_token) -> (!air.async.token) {
        %2 = scf.if %cond -> (!air.async.token) {
          %3 = air.channel.get async [%dep]  @channel_A[%c0, %c0, %arg2] (%results[] [] []) {id = 4 : i32} : (memref<64x64xbf16, 2 : i32>)
          scf.yield %3 : !air.async.token
        } else {
          %3 = air.channel.get async [%dep]  @channel_B[%c0, %c0, %arg2] (%results[] [] []) {id = 5 : i32} : (memref<64x64xbf16, 2 : i32>)
          scf.yield %3 : !air.async.token
        }
        %4 = air.channel.put async [%2]  @channel_out[%arg2, %c0] (%results_1[] [] []) {id = 6 : i32} : (memref<64x64xbf16, 2 : i32>)
        scf.yield %4 : !air.async.token
      }
      %async_token_2 = air.execute [%1] {
        memref.dealloc %results : memref<64x64xbf16, 2 : i32>
      } {id = 7 : i32}
      %async_token_3 = air.execute [%1] {
        memref.dealloc %results_1 : memref<64x64xbf16, 2 : i32>
      } {id = 8 : i32}
    }
    return
  }
}
