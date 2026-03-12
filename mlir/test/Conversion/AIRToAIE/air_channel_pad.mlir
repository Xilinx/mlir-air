//===- air_channel_pad.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Test that padding attributes on air.channel.put propagate to aie.dma_bd
// as const_pad_before/const_pad_after in the memtile DMA.

// CHECK: aie.device
// CHECK:         %[[TILE_L2:.*]] = aie.tile(2, 1)
// CHECK:         %[[TILE_L1:.*]] = aie.tile(2, 3)

// CHECK:       aie.memtile_dma(%[[TILE_L2]])
// The MM2S DMA BD from memtile to compute tile should have padding
// CHECK:             aie.dma_bd({{.*}}, 0, 16,
// CHECK-SAME:            [<size = 13, stride = 1>],
// CHECK-SAME:            [<const_pad_before = 2, const_pad_after = 1>])

module {
  air.channel @L3ToL2 [1, 1]
  air.channel @L2ToL1 [1, 1]
  func.func @pad_test(%arg0: memref<16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<16xi32> {
      air.channel.put @L3ToL2[] (%a[] [] []) : (memref<16xi32>)
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c13_s = arith.constant 13 : index
        %alloc_l2 = memref.alloc() : memref<16xi32, 1>
        air.channel.get @L3ToL2[] (%alloc_l2[] [] []) : (memref<16xi32, 1>)
        air.channel.put @L2ToL1[] (%alloc_l2[%c0_s] [%c13_s] [%c1_s])
            {pad_before = array<i32: 2>, pad_after = array<i32: 1>}
            : (memref<16xi32, 1>)
        air.herd @herd_0 tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c1_s) {
          %alloc_l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.get @L2ToL1[] (%alloc_l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %alloc_l1 : memref<16xi32, 2>
          air.herd_terminator
        }
        memref.dealloc %alloc_l2 : memref<16xi32, 1>
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
