//===- air_channel_inner_stride_offset.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" | FileCheck %s

// Each offset of an access pattern is in units of its stride, the innermost
// one included: [4] [8] [2] reads elements 8, 10, ..., 22, so the BD starts at
// element 8. In two dims, [1, 3] [2, 4] [16, 2] starts at 1 * 16 + 3 * 2 = 22.

// CHECK:       aie.memtile_dma
// CHECK-DAG:     aie.dma_bd({{.*}} : memref<64xi32, 1> offset = 8 len = 8 sizes = [8] strides = [2])
// CHECK-DAG:     aie.dma_bd({{.*}} : memref<64xi32, 1> offset = 22 len = 8 sizes = [2, 4] strides = [16, 2])

module {
  air.channel @L3ToL2 [1, 1]
  air.channel @L2ToL1 [1, 2]
  func.func @inner_stride_offset(%arg0: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<64xi32> {
      air.channel.put @L3ToL2[] (%a[] [] []) : (memref<64xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        %alloc_l2 = memref.alloc() : memref<64xi32, 1>
        air.channel.get @L3ToL2[] (%alloc_l2[] [] []) : (memref<64xi32, 1>)
        %c0_s = arith.constant 0 : index
        %c2_s = arith.constant 2 : index
        air.channel.put @L2ToL1[%c0_s, %c0_s] (%alloc_l2[4] [8] [2]) : (memref<64xi32, 1>)
        air.channel.put @L2ToL1[%c0_s, %c1_s] (%alloc_l2[1, 3] [2, 4] [16, 2]) : (memref<64xi32, 1>)
        air.herd @herd_0 tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c2_s) {
          %alloc_l1 = memref.alloc() : memref<8xi32, 2>
          air.channel.get @L2ToL1[%hx, %hy] (%alloc_l1[] [] []) : (memref<8xi32, 2>)
          memref.dealloc %alloc_l1 : memref<8xi32, 2>
        }
        memref.dealloc %alloc_l2 : memref<64xi32, 1>
      }
    }
    return
  }
}
