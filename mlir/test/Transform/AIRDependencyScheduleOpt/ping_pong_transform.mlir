//===- ping_pong_transform.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-ping-pong-transform | FileCheck %s

// Transform scf.for loops into AIR's ping-pong pattern.

// CHECK-LABEL: single_ping_pong
// CHECK: %alloc = memref.alloc() : memref<1x256x112x4xi8, 1>
// CHECK: %alloc = memref.alloc() : memref<1x256x112x4xi8, 1>
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]] = air.channel.get async [%[[EVENT4]], %[[EVENT1]]] @channel_0[]
// CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT3]], %[[EVENT5]]]
// CHECK: %[[EVENT7:.*]] = scf.for {{.*}} iter_args(%[[EVENT8:.*]] = %[[EVENT6]])
// CHECK: %[[EVENT9:.*]] = air.channel.put async [%[[EVENT8]]]  @channel_1
// CHECK: scf.yield %[[EVENT9]]
// CHECK: }
// CHECK: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK: %[[EVENT11:.*]] = air.channel.get async [%[[EVENT5]], %[[EVENT2]]] @channel_0[]
// CHECK: %[[EVENT12:.*]] = air.wait_all async [%[[EVENT10]], %[[EVENT11]]]
// CHECK: %[[EVENT13:.*]] = scf.for {{.*}} iter_args(%[[EVENT14:.*]] = %[[EVENT12]])
// CHECK: %[[EVENT15:.*]] = air.channel.put async [%[[EVENT14]]]  @channel_1
// CHECK: scf.yield %[[EVENT15]]
// CHECK: }
// CHECK: %[[EVENT16:.*]] = air.wait_all async [%[[EVENT13]]]
// CHECK: scf.yield %[[EVENT10]], %[[EVENT16]], %[[EVENT16]], %[[EVENT11]] : !air.async.token, !air.async.token, !air.async.token, !air.async.token
// CHECK: memref.dealloc {{.*}} : memref<1x256x112x4xi8, 1>
// CHECK: memref.dealloc {{.*}} : memref<1x256x112x4xi8, 1>

module {
  func.func @single_ping_pong() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async  attributes {id = 2 : i32} {
        %c448 = arith.constant 448 : index
        %c114688 = arith.constant 114688 : index
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c12544 = arith.constant 12544 : index
        %2 = air.wait_all async 
        %3 = scf.for %arg4 = %c0 to %c112 step %c4 iter_args(%arg5 = %2) -> (!air.async.token) {
          %async_token, %results = air.execute [%arg5] -> (memref<1x256x112x4xi8, 1>) {
            %alloc = memref.alloc() {hoist_alloc = "true"} : memref<1x256x112x4xi8, 1>
            air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
          }
          %5 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<1x256x112x4xi8, 1>)
          %6 = scf.for %arg6 = %c0 to %c4 step %c1_0 iter_args(%arg7 = %5) -> (!air.async.token) {
            %7 = air.channel.put async [%arg7]  @channel_1[%c0, %c0] (%results[%c0, %c0, %c0, %arg6] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 3 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %7 : !air.async.token
          }
          %async_token_1 = air.execute [%6] {
            memref.dealloc %results : memref<1x256x112x4xi8, 1>
          }
          scf.yield %async_token_1 : !air.async.token
        } {unroll = 2 : i32}
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// CHECK-LABEL: multiple_ping_pong
// CHECK: %[[EVENT0:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}} %[[EVENT2:.*]] = {{.*}} %[[EVENT3:.*]] = {{.*}} %[[EVENT4:.*]] = {{.*}})
// CHECK: %[[EVENT5:.*]]:4 = scf.for {{.*}} iter_args(%[[EVENT6:.*]] = {{.*}} %[[EVENT7:.*]] = {{.*}} %[[EVENT8:.*]] = {{.*}} %[[EVENT9:.*]] = {{.*}})

module {
  func.func @multiple_ping_pong() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async  attributes {id = 2 : i32} {
        %c448 = arith.constant 448 : index
        %c114688 = arith.constant 114688 : index
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c112 = arith.constant 112 : index
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c12544 = arith.constant 12544 : index
        %2 = air.wait_all async 
        %3 = scf.for %arg4 = %c0 to %c112 step %c4 iter_args(%arg5 = %2) -> (!air.async.token) {
          %async_token, %results = air.execute [%arg5] -> (memref<1x256x112x4xi8, 1>) {
            %alloc = memref.alloc() {hoist_alloc = "true"} : memref<1x256x112x4xi8, 1>
            air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
          }
          %5 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<1x256x112x4xi8, 1>)
          %6 = scf.for %arg6 = %c0 to %c4 step %c1_0 iter_args(%arg7 = %5) -> (!air.async.token) {
            %7 = air.channel.put async [%arg7]  @channel_1[%c0, %c0] (%results[%c0, %c0, %c0, %arg6] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 3 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %7 : !air.async.token
          }
          %async_token_1 = air.execute [%6] {
            memref.dealloc %results : memref<1x256x112x4xi8, 1>
          }
          scf.yield %async_token_1 : !air.async.token
        } {unroll = 2 : i32}
        %60 = air.wait_all async 
        %71 = scf.for %arg8 = %c0 to %c112 step %c4 iter_args(%arg9 = %60) -> (!air.async.token) {
          %async_token_200, %results_200 = air.execute [%arg9] -> (memref<1x256x112x4xi8, 1>) {
            %alloc = memref.alloc() {hoist_alloc = "true"} : memref<1x256x112x4xi8, 1>
            air.execute_terminator %alloc : memref<1x256x112x4xi8, 1>
          }
          %12 = air.channel.get async [%async_token_200]  @channel_8[] (%results_200[] [] []) {id = 2 : i32} : (memref<1x256x112x4xi8, 1>)
          %14 = scf.for %arg10 = %c0 to %c4 step %c1_0 iter_args(%arg11 = %12) -> (!air.async.token) {
            %17 = air.channel.put async [%arg11]  @channel_6[%c0, %c0] (%results_200[%c0, %c0, %c0, %arg10] [%c1_0, %c256, %c112, %c1_0] [%c114688, %c12544, %c448, %c1_0]) {id = 4 : i32} : (memref<1x256x112x4xi8, 1>)
            scf.yield %17 : !air.async.token
          }
          %async_token_dealloc_200 = air.execute [%14] {
            memref.dealloc %results_200 : memref<1x256x112x4xi8, 1>
          }
          scf.yield %async_token_dealloc_200 : !air.async.token
        } {unroll = 2 : i32}
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
