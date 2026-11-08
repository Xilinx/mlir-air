//===- air_specialize_channel_broadcast_1d.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// A broadcast to a one-dimensional herd: the herd has no row extent to read.

// CHECK: air.channel @Bcast [1] {broadcast_shape = [4]}
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@Bcast

module {
  air.channel @Bcast [1] {broadcast_shape = [4]}
  func.func @one_dim(%arg0: memref<64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg5, %arg6) in (%arg7=%c1, %arg8=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async  attributes {id = 1 : i32} {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64xbf16, 1 : i32>
        } {id = 1 : i32}
        %2 = air.channel.put async [%async_token]  @Bcast[] (%results[] [] []) {id = 1 : i32} : (memref<64xbf16, 1 : i32>)
        %async_token_1, %results_2 = air.execute -> (memref<64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
        } {id = 2 : i32}
        %4 = air.herd @herd_0 async [%async_token_1]  tile (%arg9) in (%arg11=%c4) args(%arg13=%results_2) : memref<64xbf16, 2 : i32> attributes {id = 1 : i32} {
          %5 = air.channel.get async  @Bcast[%arg9] (%arg13[] [] []) {id = 3 : i32} : (memref<64xbf16, 2 : i32>)
        }
      }
    }
    return
  }
}
