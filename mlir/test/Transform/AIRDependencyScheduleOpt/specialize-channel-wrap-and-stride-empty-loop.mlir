//===- specialize-channel-wrap-and-stride-empty-loop.mlir ------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride="scope=func" | FileCheck %s

// A loop that never runs has a trip count of 0. Folding an arith.muli or
// arith.addi on its induction variable into its bounds divides by the trip
// count, so it must be skipped.

// CHECK-LABEL: func.func @empty_loop_muli
// CHECK-LABEL: func.func @empty_loop_addi
module {
  air.channel @channel_0 [1, 1]
  func.func @empty_loop_muli(%arg0: memref<32x16xi32>) {
    %0 = air.launch async () in () args(%arg2=%arg0) : memref<32x16xi32> {
      %1 = air.segment @seg async  args(%arg3=%arg2) : memref<32x16xi32> {
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %2 = air.wait_all async
        %3 = scf.for %arg4 = %c2 to %c0 step %c1 iter_args(%arg5 = %2) -> (!air.async.token) {
          %async_token, %results = air.execute [%arg5] -> (index) {
            %6 = arith.muli %arg4, %c16 : index
            air.execute_terminator %6 : index
          }
          %5 = air.channel.put async [%async_token, %arg5]  @channel_0[%c0, %c0] (%arg3[%c0, %results] [%c16, %c8] [%c8, %c1]) : (memref<32x16xi32>)
          scf.yield %5 : !air.async.token
        }
      }
    }
    return
  }
  func.func @empty_loop_addi(%arg0: memref<32x16xi32>) {
    %0 = air.launch async () in () args(%arg2=%arg0) : memref<32x16xi32> {
      %1 = air.segment @seg async  args(%arg3=%arg2) : memref<32x16xi32> {
        %c16 = arith.constant 16 : index
        %c8 = arith.constant 8 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %2 = air.wait_all async
        %3 = scf.for %arg4 = %c2 to %c0 step %c1 iter_args(%arg5 = %2) -> (!air.async.token) {
          %async_token, %results = air.execute [%arg5] -> (index) {
            %6 = arith.addi %arg4, %c1 : index
            air.execute_terminator %6 : index
          }
          %5 = air.channel.put async [%async_token, %arg5]  @channel_0[%c0, %c0] (%arg3[%results, %c0] [%c16, %c8] [%c8, %c1]) : (memref<32x16xi32>)
          scf.yield %5 : !air.async.token
        }
      }
    }
    return
  }
}
