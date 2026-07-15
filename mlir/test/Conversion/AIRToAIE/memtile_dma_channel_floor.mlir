//===- memtile_dma_channel_floor.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=1 num-cols=1 row-anchor=3 col-anchor=5' -air-to-aie="device=xcve2802" | FileCheck %s

// MemTile DMA channel floor. A memtile DMA memcpy carrying
// `air.memtile_dma_channel_min = N` reserves physical channels [0, N) on that
// memtile, so its flow lands on [N, ...). Here the L2->L1 feed (@channel_1)
// would otherwise take MM2S 0; the floor moves it to MM2S 1 (while the L3->L2
// S2MM keeps channel 0). Used when a flow's route on a low physical channel
// collides with another column flow transiting the memtile switchbox.

// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 1
air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func @memtile_dma_channel_floor(%arg0: memref<32xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%arg1) in (%arg3=%c1) args(%arg5=%arg0) : memref<32xi32> {
    %1 = air.channel.put async @channel_0[] (%arg5[] [] []) : (memref<32xi32>)
    %2 = air.segment async {
      %async_token_1, %results_1 = air.execute -> (memref<32xi32, 1>) {
        %alloc1 = memref.alloc() : memref<32xi32, 1>
        air.execute_terminator %alloc1 : memref<32xi32, 1>
      }
      %4 = air.channel.get async [%async_token_1] @channel_0[] (%results_1[] [] []) : (memref<32xi32, 1>)
      %6 = air.channel.put async [%4] @channel_1[] (%results_1[] [] []) {air.memtile_dma_channel_min = 1 : i32} : (memref<32xi32, 1>)
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
