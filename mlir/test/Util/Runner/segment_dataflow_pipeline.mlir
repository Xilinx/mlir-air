//===- segment_dataflow_pipeline.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// A dataflow pipeline made of air.segments as pipeline stages.
// Stages are connected with FIFOs to enable concurrent execution.

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
  air.channel @channel_3 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @test() {
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
          %6 = air.channel.put async [%async_token]  @channel_0[] (%results[] [] []) {id = 16 : i32} : (memref<128x128xbf16, 1>)
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
        } {ping_pong = 0 : ui32}
        %async_token_1, %results_2 = air.execute [%4] -> (memref<128x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x1024xbf16, 1>
          air.execute_terminator %alloc : memref<128x1024xbf16, 1>
        } {ping_pong = 1 : ui32}
        %5:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %async_token, %arg6 = %async_token_1, %arg7 = %async_token_1, %arg8 = %async_token_1) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %8 = air.channel.get async [%arg8, %arg5]  @channel_0[] (%results[%arg4, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, ping_pong = 0 : ui32, producer = 0 : ui32} : (memref<128x1024xbf16, 1>)
          %9 = air.wait_all async [%arg7, %8] 
          %10 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %9) -> (!air.async.token) {
            %15 = air.channel.put async [%arg10]  @channel_1[] (%results[%c0, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
            scf.yield %15 : !air.async.token
          } {async_back = true, consumer = 0 : ui32, ping_pong = 0 : ui32}
          %waitall_1 = air.wait_all async [%10]
          %11 = arith.addi %arg4, %c256 : index
          %12 = air.channel.get async [%8, %arg6]  @channel_0[] (%results_2[%11, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, ping_pong = 1 : ui32, producer = 1 : ui32} : (memref<128x1024xbf16, 1>)
          %13 = air.wait_all async [%waitall_1, %12] 
          %14 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %13) -> (!air.async.token) {
            %15 = air.channel.put async [%arg10]  @channel_1[] (%results_2[%c0, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
            scf.yield %15 : !air.async.token
          } {async_back = true, consumer = 1 : ui32, ping_pong = 1 : ui32}
          scf.yield %waitall_1, %14, %14, %12 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_3 = air.execute [%5#1] {
          memref.dealloc %results : memref<128x1024xbf16, 1>
        } {ping_pong = 0 : ui32}
        %async_token_4 = air.execute [%5#1] {
          memref.dealloc %results_2 : memref<128x1024xbf16, 1>
        } {ping_pong = 1 : ui32}
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
            %10 = air.channel.get async [%async_token_8]  @channel_1[] (%results_9[] [] []) {id = 22 : i32} : (memref<128x128xbf16, 1>)
            %async_token_10 = air.execute [%10] {
              memref.dealloc %results_9 : memref<128x128xbf16, 1>
            }
            scf.yield %async_token_10 : !air.async.token
          }
          %9 = air.channel.put async [%8]  @channel_2[] (%results_6[] [] []) {id = 32 : i32} : (memref<128x128xbf16, 1>)
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
        } {ping_pong = 0 : ui32}
        %async_token_1, %results_2 = air.execute [%4] -> (memref<128x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x1024xbf16, 1>
          air.execute_terminator %alloc : memref<128x1024xbf16, 1>
        } {ping_pong = 1 : ui32}
        %5:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %async_token, %arg6 = %async_token_1, %arg7 = %async_token_1, %arg8 = %async_token_1) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %8 = air.channel.get async [%arg8, %arg5]  @channel_2[] (%results[%arg4, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, ping_pong = 0 : ui32, producer = 0 : ui32} : (memref<128x1024xbf16, 1>)
          %9 = air.wait_all async [%arg7, %8] 
          %10 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %9) -> (!air.async.token) {
            %15 = air.channel.put async [%arg10]  @channel_3[] (%results[%arg4, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
            scf.yield %15 : !air.async.token
          } {async_back = true, consumer = 0 : ui32, ping_pong = 0 : ui32}
          %waitall_2 = air.wait_all async [%10]
          %11 = arith.addi %arg4, %c256 : index
          %12 = air.channel.get async [%8, %arg6]  @channel_2[] (%results_2[%11, %c0] [%c128, %c128] [%c1024, %c1_0]) {async_front = true, id = 4 : i32, ping_pong = 1 : ui32, producer = 1 : ui32} : (memref<128x1024xbf16, 1>)
          %13 = air.wait_all async [%waitall_2, %12] 
          %14 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %13) -> (!air.async.token) {
            %15 = air.channel.put async [%arg10]  @channel_3[] (%results_2[%11, %c0] [%c128, %c128] [%c128, %c1_0]) {id = 18 : i32} : (memref<128x1024xbf16, 1>)
            scf.yield %15 : !air.async.token
          } {async_back = true, consumer = 1 : ui32, ping_pong = 1 : ui32}
          scf.yield %waitall_2, %14, %14, %12 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_3 = air.execute [%5#1] {
          memref.dealloc %results : memref<128x1024xbf16, 1>
        } {ping_pong = 0 : ui32}
        %async_token_4 = air.execute [%5#1] {
          memref.dealloc %results_2 : memref<128x1024xbf16, 1>
        } {ping_pong = 1 : ui32}
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
            %9 = air.channel.get async [%async_token_8]  @channel_3[] (%results_9[] [] []) {id = 38 : i32} : (memref<128x128xbf16, 1>)
            %async_token_10 = air.execute [%9] {
              memref.dealloc %results_9 : memref<128x128xbf16, 1>
            }
            scf.yield %async_token_10 : !air.async.token
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
}
