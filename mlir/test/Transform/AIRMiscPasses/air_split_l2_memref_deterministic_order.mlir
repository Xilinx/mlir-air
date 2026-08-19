//===- air_split_l2_memref_deterministic_order.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// Two splittable L2 buffers in one func. The pass mints one fresh channel name
// per buffer, so the name each buffer gets is decided by the order the pass
// walks them. That order used to come from a DenseMap keyed by the alloc op,
// i.e. from heap addresses, so the two buffers swapped names between processes
// -- and since the names drive aie.shim_dma_allocation, so did the physical
// shim port assignment.
//
// The two buffers split by different factors (2 and 4), so the minted sizes
// tell them apart: walk order pins @channel_0 (from the 2-way buffer, declared
// first) ahead of @channel_1 (the 4-way one). A single alloc cannot catch this
// -- one entry has no order to get wrong.
// CHECK: air.channel @channel_0 [2, 1]
// CHECK: air.channel @channel_1 [4, 1]
// CHECK-LABEL: func.func @two_allocs_keep_walk_order
// CHECK: memref.alloc() : memref<2560xbf16, 1 : i32>
// CHECK: memref.alloc() : memref<3072xbf16, 1 : i32>
// CHECK-NOT: memref<5120xbf16, 1 : i32>
// CHECK-NOT: memref<12288xbf16, 1 : i32>

air.channel @inA [1]
air.channel @inB [1]
air.channel @aL2ToL1 [1, 2]
air.channel @bL2ToL1 [1, 4]
func.func @two_allocs_keep_walk_order(%arg0: memref<5120xbf16>, %arg1: memref<12288xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg2) in (%arg3=%c1) args(%arg4=%arg0, %arg5=%arg1) : memref<5120xbf16>, memref<12288xbf16> attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      // Buffer A: 2 destination tiles -> split factor 2.
      %async_token, %results = air.execute -> (memref<5120xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<5120xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<5120xbf16, 1 : i32>
      }
      %2 = air.channel.get async [%async_token]  @inA[%c0] (%results[] [] []) {id = 1 : i32} : (memref<5120xbf16, 1 : i32>)
      %3 = air.channel.put async [%2]  @aL2ToL1[%c0, %c0] (%results[0] [2560] [1]) {id = 2 : i32} : (memref<5120xbf16, 1 : i32>)
      %4 = air.channel.put async [%2]  @aL2ToL1[%c0, %c1_0] (%results[2560] [2560] [1]) {id = 3 : i32} : (memref<5120xbf16, 1 : i32>)
      %async_token_0 = air.execute [%3, %4] {
        memref.dealloc %results : memref<5120xbf16, 1 : i32>
      }
      // Buffer B: 4 destination tiles -> split factor 4.
      %async_token_1, %results_1 = air.execute -> (memref<12288xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<12288xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<12288xbf16, 1 : i32>
      }
      %5 = air.channel.get async [%async_token_1]  @inB[%c0] (%results_1[] [] []) {id = 4 : i32} : (memref<12288xbf16, 1 : i32>)
      %6 = air.channel.put async [%5]  @bL2ToL1[%c0, %c0] (%results_1[0] [3072] [1]) {id = 5 : i32} : (memref<12288xbf16, 1 : i32>)
      %7 = air.channel.put async [%5]  @bL2ToL1[%c0, %c1_0] (%results_1[3072] [3072] [1]) {id = 6 : i32} : (memref<12288xbf16, 1 : i32>)
      %8 = air.channel.put async [%5]  @bL2ToL1[%c0, %c2] (%results_1[6144] [3072] [1]) {id = 7 : i32} : (memref<12288xbf16, 1 : i32>)
      %9 = air.channel.put async [%5]  @bL2ToL1[%c0, %c3] (%results_1[9216] [3072] [1]) {id = 8 : i32} : (memref<12288xbf16, 1 : i32>)
      %async_token_2 = air.execute [%6, %7, %8, %9] {
        memref.dealloc %results_1 : memref<12288xbf16, 1 : i32>
      }
      %10 = air.herd @herd_a async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c2) attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %async_token_3, %results_3 = air.execute -> (memref<2560xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<2560xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<2560xbf16, 2 : i32>
        }
        %11 = air.channel.get async [%async_token_3]  @aL2ToL1[%arg6, %arg7] (%results_3[] [] []) {id = 9 : i32} : (memref<2560xbf16, 2 : i32>)
        %async_token_4 = air.execute [%11] {
          memref.dealloc %results_3 : memref<2560xbf16, 2 : i32>
        }
      }
      %12 = air.herd @herd_b async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 4 : i64} {
        %async_token_5, %results_5 = air.execute -> (memref<3072xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<3072xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<3072xbf16, 2 : i32>
        }
        %13 = air.channel.get async [%async_token_5]  @bL2ToL1[%arg6, %arg7] (%results_5[] [] []) {id = 10 : i32} : (memref<3072xbf16, 2 : i32>)
        %async_token_6 = air.execute [%13] {
          memref.dealloc %results_5 : memref<3072xbf16, 2 : i32>
        }
      }
    }
  }
  return
}
