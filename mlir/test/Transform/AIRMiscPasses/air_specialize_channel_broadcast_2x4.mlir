//===- air_specialize_channel_broadcast_2x4.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// A broadcast over each column of a 2 x 4 herd. The row a core may take
// ranges over the herd's 4 rows, not its 2 columns.

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
// CHECK: air.channel @ColBcast_0 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @ColBcast_1 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.herd @herd_0
// CHECK: affine.if [[$SET0]]
// CHECK: @ColBcast_0
// CHECK: } else {
// CHECK: @ColBcast_1

module {
  air.channel @ColBcast [2, 1] {broadcast_shape = [2, 4]}
  func.func @two_by_four(%arg0: memref<64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg5, %arg6) in (%arg7=%c1, %arg8=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async  attributes {id = 1 : i32} {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64xbf16, 1 : i32>
        } {id = 1 : i32}
        %2 = air.channel.put async [%async_token]  @ColBcast[%c0, %c0] (%results[] [] []) {id = 1 : i32} : (memref<64xbf16, 1 : i32>)
        %3 = air.channel.put async [%async_token]  @ColBcast[%c1_0, %c0] (%results[] [] []) {id = 2 : i32} : (memref<64xbf16, 1 : i32>)
        %async_token_1, %results_2 = air.execute -> (memref<64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
        } {id = 2 : i32}
        %4 = air.herd @herd_0 async [%async_token_1]  tile (%arg9, %arg10) in (%arg11=%c2, %arg12=%c4) args(%arg13=%results_2) : memref<64xbf16, 2 : i32> attributes {id = 1 : i32} {
          %5 = air.channel.get async  @ColBcast[%arg9, %arg10] (%arg13[] [] []) {id = 3 : i32} : (memref<64xbf16, 2 : i32>)
        }
      }
    }
    return
  }
}
