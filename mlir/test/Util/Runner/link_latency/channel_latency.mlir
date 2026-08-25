//===- channel_latency.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A single L2->L1 channel transfer over a link that carries a fixed
// time-of-flight in addition to its bandwidth cost.
//
// arch.json gives the L2 outbound port 300 cycles of latency and the L1
// inbound port 200, so the L2->L1 interface latency is 300 + 200 = 500 cycles.
// Both ports run at 1 byte/cycle, so a memref<64xi8> occupies the link for 64
// cycles. The get therefore completes 64 + 500 = 564 cycles after it starts.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK: "ts": 0.0[[#%d,TIME0:]],
// CHECK: "name": "ChannelGetOp@channel_0(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK: "ts": 0.[[#TIME0 + 564]],

module {
  air.channel @channel_0 [1, 1]
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<64xi8, 1>) {
          %alloc = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %alloc : memref<64xi8, 1>
        }
        %2 = air.channel.put async [%async_token]  @channel_0[] (%results[] [] []) {id = 3 : i32} : (memref<64xi8, 1>)
        %3 = air.herd @herd_0 async tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_0) attributes {id = 4 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %async_token_1, %results_2 = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %4 = air.channel.get async [%async_token_1]  @channel_0[] (%results_2[] [] []) {id = 5 : i32} : (memref<64xi8, 2>)
          %async_token_3 = air.execute [%4] {
            memref.dealloc %results_2 : memref<64xi8, 2>
          }
        }
        %async_token_4 = air.execute [%2] {
          memref.dealloc %results : memref<64xi8, 1>
        }
      }
    }
    return
  }
}
