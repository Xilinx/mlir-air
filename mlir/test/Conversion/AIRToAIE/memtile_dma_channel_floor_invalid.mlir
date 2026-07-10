//===- memtile_dma_channel_floor_invalid.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=1 num-cols=1 row-anchor=3 col-anchor=5' -air-to-aie="device=xcve2802" -verify-diagnostics -split-input-file

// An `air.memtile_dma_channel_min` floor at or beyond the memtile's available
// DMA-channel count leaves no usable channel; it must be a hard error rather
// than a silent fallback to round-robin. A memtile has 6 MM2S DMA channels, so
// a floor of 9 is rejected.

air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func @memtile_dma_channel_floor_out_of_range(%arg0: memref<32xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%arg1) in (%arg3=%c1) args(%arg5=%arg0) : memref<32xi32> {
    %1 = air.channel.put async @channel_0[] (%arg5[] [] []) : (memref<32xi32>)
    %2 = air.segment async {
      %async_token_1, %results_1 = air.execute -> (memref<32xi32, 1>) {
        %alloc1 = memref.alloc() : memref<32xi32, 1>
        air.execute_terminator %alloc1 : memref<32xi32, 1>
      }
      %4 = air.channel.get async [%async_token_1] @channel_0[] (%results_1[] [] []) : (memref<32xi32, 1>)
      // expected-error @+1 {{air.memtile_dma_channel_min = 9 is out of range [0, 6) for the MM2S DMA channels of this memtile}}
      %6 = air.channel.put async [%4] @channel_1[] (%results_1[] [] []) {air.memtile_dma_channel_min = 9 : i32} : (memref<32xi32, 1>)
      %c1_2 = arith.constant 1 : index
      %7 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_2, %arg11=%c1_2) {
        %async_token_2, %results_2 = air.execute -> (memref<32xi32, 2>) {
          %alloc2 = memref.alloc() : memref<32xi32, 2>
          air.execute_terminator %alloc2 : memref<32xi32, 2>
        }
        %9 = air.channel.get async [%async_token_2] @channel_1[] (%results_2[] [] []) : (memref<32xi32, 2>)
        %async_token_3 = air.execute [%9] {
          memref.dealloc %results_2 : memref<32xi32, 2>
        }
      }
    }
  }
  return
}
