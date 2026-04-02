//===- split_launch_for_padding_channel_source.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding='pad-location=source' | FileCheck %s

// Tests source-level padding with channel ops (combination 4: GPU + channels).
// M=500, M_TILE=64 -> launchM=8, last M-block has 500-7*64=52 rows
// N=500, N_TILE=32 -> launchN=16, last N-block has 500-15*32=20 cols
// Expected: 4 partitions (interior 7x15, m_boundary 1x15, n_boundary 7x1, corner 1x1)

// CHECK-LABEL: func.func @channel_source_padding

// Interior: no padding.
// CHECK: air.segment @channel_source_padding_0_interior
// CHECK: air.channel.put @chanA
// CHECK-NOT: pad_after
// CHECK: air.channel.put @chanB
// CHECK-NOT: pad_after

// M-boundary: A channel.put has reduced sizes and pad_after.
// CHECK: air.segment @channel_source_padding_0_m_boundary
// CHECK: air.channel.put @chanA{{.*}}pad_after = array<i32: 12, 0>, pad_before = array<i32: 0, 0>
// CHECK: air.channel.put @chanB
// CHECK-NOT: pad_after

// N-boundary: B channel.put has reduced sizes and pad_after.
// CHECK: air.segment @channel_source_padding_0_n_boundary
// CHECK: air.channel.put @chanA
// CHECK-NOT: pad_after
// CHECK: air.channel.put @chanB{{.*}}pad_after = array<i32: 0, 12>, pad_before = array<i32: 0, 0>

// Corner: both A and B have padding.
// CHECK: air.segment @channel_source_padding_0_corner
// CHECK: air.channel.put @chanA{{.*}}pad_after = array<i32: 12, 0>, pad_before = array<i32: 0, 0>
// CHECK: air.channel.put @chanB{{.*}}pad_after = array<i32: 0, 12>, pad_before = array<i32: 0, 0>

module {
  air.channel @chanA []
  air.channel @chanB []
  func.func @channel_source_padding(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c8, %arg13=%c16, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {air.actual_sizes = array<i64: 500, 500, 1>} {
      air.segment @channel_source_padding_0  args(%arg18=%arg9, %arg19=%arg10, %arg20=%arg15, %arg21=%arg16, %arg22=%arg17) : index, index, memref<*xbf16>, memref<*xbf16>, memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c16_1 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c500 = arith.constant 500 : index
        %2 = arith.muli %arg18, %c64 : index
        %3 = arith.muli %arg19, %c32 : index
        %10 = arith.muli %arg18, %c500 : index
        %11 = arith.addi %2, %10 : index
        // A: src_sizes = [64, 16], tile size 64 is in dim 0
        air.channel.put @chanA[](%arg20[%c0, %11] [%c64, %c16_1] [%c1_0, %c500]) : (memref<*xbf16>)
        %13 = arith.addi %10, %3 : index
        // B: src_sizes = [16, 32], tile size 32 is in dim 1
        air.channel.put @chanB[](%arg21[%c0, %13] [%c16_1, %c32] [%c500, %c1_0]) : (memref<*xbf16>)
      }
    }
    return
  }
}
