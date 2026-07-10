//===- air_split_l2_memref_nonchannel_user.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" | FileCheck %s

// Precondition: the L2-memref split only redistributes channel put/get uses
// (and the dealloc) onto sub-buffers; it has no way to reassign any other use.
// Here the 8x1024 buffer is captured as an air.herd operand in addition to its
// multiple-in-single-out channel pattern. Because that operand cannot be
// mapped to a single sub-buffer, the buffer is not partitionable and must be
// left intact (previously the pass split the channels and then asserted while
// erasing the still-referenced alloc).

// CHECK-LABEL: func.func @nonchannel_user
// CHECK: memref.alloc() : memref<8x1024xbf16, 1 : i32>
// CHECK-NOT: memref<8x512xbf16, 1 : i32>

air.channel @chan_in [4, 2]
air.channel @chan_out [4]
func.func @nonchannel_user(%arg0: memref<128x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg1) in (%arg2=%c1) args(%arg3=%arg0) : memref<128x1024xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %g0 = air.channel.get async @chan_out[%c0] (%arg3[%c0, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 1 : i32} : (memref<128x1024xbf16>)
    %5 = air.segment @seg async {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %c4_0 = arith.constant 4 : index
      %c8_0 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c1024_0 = arith.constant 1024 : index
      %async_token_a, %a0 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      // Multiple-in-single-out channel pattern (would otherwise split).
      %p00 = air.channel.get async [%async_token_a] @chan_in[%c0_0, %c0_0] (%a0[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p01 = air.channel.get async [%async_token_a] @chan_in[%c0_0, %c1_0] (%a0[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %op0 = air.channel.put async [%p00, %p01] @chan_out[%c0_0] (%a0[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      // Non-channel use: the L2 buffer is captured as an air.herd operand.
      %h = air.herd @h async tile (%tx, %ty) in (%sx=%c4_0, %sy=%c2_0) args(%a=%a0) : memref<8x1024xbf16, 1 : i32> {
        %async_token_h, %r = air.execute -> (memref<8x512xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<8x512xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<8x512xbf16, 2 : i32>
        }
        %ph = air.channel.put async [%async_token_h] @chan_in[%tx, %ty] (%r[] [] []) : (memref<8x512xbf16, 2 : i32>)
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
