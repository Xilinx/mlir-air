//===- affine_if_filtered_ops_retire.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/../arch.json -g core | FileCheck %s

// A broadcast pair guarded by affine.if, simulated per core.
//
// Each core takes exactly one side of the branch, so at core granularity the
// other side is filtered out of scheduling. Filtered ops have to be retired,
// not merely skipped: an op left unstarted blocks every vertex that lists it
// as a dependence, starting with its own branch terminator.
//
// The assertion here is not the cycle count -- retiring costs no time, so the
// count is the same either way. It is that the run leaves nothing behind.
// air-runner fails a run that reaches the launch terminator with ops that
// never ran and were never retired, so reaching this CHECK at all is the test:
// without the retirement each of the four cores strands its unselected branch
// and the run is diagnosed instead of reporting a latency.

// CHECK: "name": "LaunchTerminator",
// CHECK: Latency (all-iterations mode): 0.097us

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @row0 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @row1 [1, 1] {broadcast_shape = [1, 2]}
  func.func @test() {
    %0 = air.launch async () in () attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 2 : i64, y_loc = 0 : i64, y_size = 2 : i64} {
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %2 = air.channel.put async [%async_token] @row0[] (%results[] [] []) {id = 1 : i32} : (memref<64x64xbf16, 1 : i32>)
        %3 = air.channel.put async [%async_token] @row1[] (%results[] [] []) {id = 2 : i32} : (memref<64x64xbf16, 1 : i32>)
        %4 = air.herd @herd_0 async [%async_token] tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c2) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %5 = affine.if #set()[%arg0, %arg1] -> !air.async.token {
            %6 = air.channel.get async [%async_token_0] @row0[%arg0, %arg1] (%results_1[] [] []) {id = 3 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %6 : !air.async.token
          } else {
            %6 = air.channel.get async [%async_token_0] @row1[%arg0, %arg1] (%results_1[] [] []) {id = 4 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %6 : !air.async.token
          }
          %async_token_2 = air.execute [%5] {
            memref.dealloc %results_1 : memref<64x64xbf16, 2 : i32>
          }
        }
        %async_token_3 = air.execute [%4, %2, %3] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
