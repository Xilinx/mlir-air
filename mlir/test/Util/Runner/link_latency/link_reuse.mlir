//===- link_reuse.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two independent transfers contending for a single L1 inbound port, over a
// link that carries a fixed time-of-flight.
//
// This is the regression test for separating link occupancy from link latency.
// A port models bandwidth, not flight time, so it must be released once the
// payload has been clocked out (64 cycles here) rather than when the payload
// lands (64 + 500). The second transfer therefore starts at +64 and the pair
// completes in 64 + 64 + 500 = 628 cycles, not 2 * (64 + 500) = 1128.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// The first get starts, and the second claims the freed port one occupancy later.
// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK: "ts": 0.0[[#%d,TIME0:]],
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK: "ts": 0.0[[#TIME0 + 64]],

// Both land 500 cycles after their own occupancy ends.
// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK: "ts": 0.[[#TIME0 + 564]],
// CHECK: "name": "ChannelGetOp@channel_1(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK: "ts": 0.[[#TIME0 + 628]],

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<64xi8, 1>) {
          %alloc = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %alloc : memref<64xi8, 1>
        }
        %async_token_0, %results_0 = air.execute -> (memref<64xi8, 1>) {
          %alloc = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %alloc : memref<64xi8, 1>
        }
        %2 = air.channel.put async [%async_token]  @channel_0[] (%results[] [] []) {id = 3 : i32} : (memref<64xi8, 1>)
        %3 = air.channel.put async [%async_token_0]  @channel_1[] (%results_0[] [] []) {id = 4 : i32} : (memref<64xi8, 1>)
        %4 = air.herd @herd_0 async tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_0) attributes {id = 5 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %async_token_1, %results_2 = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %async_token_2, %results_3 = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %5 = air.channel.get async [%async_token_1]  @channel_0[] (%results_2[] [] []) {id = 6 : i32} : (memref<64xi8, 2>)
          %6 = air.channel.get async [%async_token_2]  @channel_1[] (%results_3[] [] []) {id = 7 : i32} : (memref<64xi8, 2>)
          %async_token_3 = air.execute [%5] {
            memref.dealloc %results_2 : memref<64xi8, 2>
          }
          %async_token_4 = air.execute [%6] {
            memref.dealloc %results_3 : memref<64xi8, 2>
          }
        }
        %async_token_5 = air.execute [%2] {
          memref.dealloc %results : memref<64xi8, 1>
        }
        %async_token_6 = air.execute [%3] {
          memref.dealloc %results_0 : memref<64xi8, 1>
        }
      }
    }
    return
  }
}
