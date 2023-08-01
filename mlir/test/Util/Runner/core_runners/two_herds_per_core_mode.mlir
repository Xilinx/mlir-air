//===- two_herds_per_core_mode.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/../arch.json -g core | FileCheck %s

// Test for core-to-core broadcast copy across two herds.


// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#set = affine_set<()[s0, s1] : (s0 == 0, s1 == 0)>
module {
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_1 [1, 1]
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
      %1 = air.segment async  attributes {x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %2 = air.herd @herd_0 async  tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %4 = air.wait_all async 
          %5 = scf.for %arg8 = %c0 to %c8 step %c4 iter_args(%arg9 = %4) -> (!air.async.token) {
            %6 = scf.for %arg10 = %c0 to %c4 step %c1_1 iter_args(%arg11 = %arg9) -> (!air.async.token) {
              %async_token, %results = air.execute -> (memref<1x32x32x1xbf16, 2>) {
                %alloc = memref.alloc() : memref<1x32x32x1xbf16, 2>
                air.execute_terminator %alloc : memref<1x32x32x1xbf16, 2>
              }
              %7 = air.channel.put async [%arg11]  @channel_0[%arg4, %arg5] (%results[] [] []) {id = 6 : i32} : (memref<1x32x32x1xbf16, 2>)
              %async_token_2 = air.execute [%7] {
                memref.dealloc %results : memref<1x32x32x1xbf16, 2>
              }
              scf.yield %7 : !air.async.token
            }
            scf.yield %6 : !air.async.token
          }
          air.herd_terminator
        }
        %3 = air.herd @herd_1 async  tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c2) attributes {id = 4 : i32, x_loc = 0 : i64, y_loc = 1 : i64} {
          %c0 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c3072 = arith.constant 3072 : index
          %c96 = arith.constant 96 : index
          %c64 = arith.constant 64 : index
          %c3 = arith.constant 3 : index
          %async_token, %results = air.execute -> (memref<1x32x32x3xbf16, 2>) {
            %alloc = memref.alloc() : memref<1x32x32x3xbf16, 2>
            air.execute_terminator %alloc : memref<1x32x32x3xbf16, 2>
          }
          %async_token_2, %results_3 = air.execute -> (memref<1x32x32x3xbf16, 2>) {
            %alloc = memref.alloc() : memref<1x32x32x3xbf16, 2>
            air.execute_terminator %alloc : memref<1x32x32x3xbf16, 2>
          }
          %4 = scf.for %arg8 = %c0 to %c8 step %c1_1 iter_args(%arg9 = %async_token) -> (!air.async.token) {
            %5 = air.channel.get async [%arg9]  @channel_0[%arg4, %arg5] (%results[%c0, %c0, %c0, %arg8] [%c1_1, %c64, %c8, %c1_1] [%c3072, %c96, %c3, %c1_1]) {id = 9 : i32} : (memref<1x32x32x3xbf16, 2>)
            %6 = affine.if #set()[%arg4, %arg5] -> !air.async.token {
              %7 = air.channel.put async [%5]  @channel_1[] (%results_3[] [] []) {id = 10 : i32} : (memref<1x32x32x3xbf16, 2>)
              affine.yield %7 : !air.async.token
            } else {
              %7 = air.channel.get async [%5]  @channel_1[] (%results_3[] [] []) {id = 10 : i32} : (memref<1x32x32x3xbf16, 2>)
              affine.yield %7 : !air.async.token
            }
            %wait_all = air.wait_all async [%6]
            scf.yield %wait_all : !air.async.token
          }
          %async_token_4 = air.execute [%4] {
            memref.dealloc %results : memref<1x32x32x3xbf16, 2>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
