//===- air_channel_pad_maxsize.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" -split-input-file 2>&1 | FileCheck %s

// A padded walk with an unpadded dimension past the 1023 limit: that
// dimension is split, and the padding stays on the dimension it was on.

// CHECK: aie.dma_bd({{.*}} : memref<4x2048xi32, 1> offset = 0 len = 8192 sizes = [3, 4, 512] strides = [2048, 512, 1]
// CHECK-SAME: pad [<const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>])

module {
  air.channel @L3ToL2 [1, 1]
  air.channel @L2ToL1 [1, 2]
  func.func @pad_maxsize(%arg0: memref<4x2048xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<4x2048xi32> {
      air.channel.put @L3ToL2[] (%a[] [] []) : (memref<4x2048xi32>)
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %alloc_l2 = memref.alloc() : memref<4x2048xi32, 1>
        air.channel.get @L3ToL2[] (%alloc_l2[] [] []) : (memref<4x2048xi32, 1>)
        air.channel.put @L2ToL1[%c0_s, %c0_s] (%alloc_l2[0, 0] [4, 2048] [2048, 1])
            : (memref<4x2048xi32, 1>)
        air.channel.put @L2ToL1[%c0_s, %c1_s] (%alloc_l2[0, 0] [3, 2048] [2048, 1])
            {pad_before = array<i32: 0, 0>, pad_after = array<i32: 1, 0>}
            : (memref<4x2048xi32, 1>)
        air.herd @herd_0 tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c2_s) {
          %alloc_l1 = memref.alloc() : memref<4x2048xi32, 2>
          air.channel.get @L2ToL1[%hx, %hy] (%alloc_l1[] [] []) : (memref<4x2048xi32, 2>)
          memref.dealloc %alloc_l1 : memref<4x2048xi32, 2>
        }
        memref.dealloc %alloc_l2 : memref<4x2048xi32, 1>
      }
    }
    return
  }
}

// -----

// A padded dimension past the limit cannot be split without moving its
// padding.

// CHECK: error: 'air.channel.put' op padded dimension 1 has 1100 elements, more than the 1023 a buffer descriptor dimension takes

module {
  air.channel @L3ToL2 [1, 1]
  air.channel @L2ToL1 [1, 2]
  func.func @pad_too_long(%arg0: memref<4x2048xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<4x2048xi32> {
      air.channel.put @L3ToL2[] (%a[] [] []) : (memref<4x2048xi32>)
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %alloc_l2 = memref.alloc() : memref<4x2048xi32, 1>
        air.channel.get @L3ToL2[] (%alloc_l2[] [] []) : (memref<4x2048xi32, 1>)
        air.channel.put @L2ToL1[%c0_s, %c0_s] (%alloc_l2[0, 0] [4, 2048] [2048, 1])
            : (memref<4x2048xi32, 1>)
        air.channel.put @L2ToL1[%c0_s, %c1_s] (%alloc_l2[0, 0] [2, 1100] [2048, 1])
            {pad_before = array<i32: 0, 0>, pad_after = array<i32: 0, 4>}
            : (memref<4x2048xi32, 1>)
        air.herd @herd_0 tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c2_s) {
          %alloc_l1 = memref.alloc() : memref<4x2048xi32, 2>
          air.channel.get @L2ToL1[%hx, %hy] (%alloc_l1[] [] []) : (memref<4x2048xi32, 2>)
          memref.dealloc %alloc_l1 : memref<4x2048xi32, 2>
        }
        memref.dealloc %alloc_l2 : memref<4x2048xi32, 1>
      }
    }
    return
  }
}
