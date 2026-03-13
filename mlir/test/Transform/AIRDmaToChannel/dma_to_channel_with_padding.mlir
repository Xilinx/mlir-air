//===- dma_to_channel_with_padding.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that padding attributes on air.dma_memcpy_nd propagate to the generated
// air.channel.put during -air-dma-to-channel conversion.

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// CHECK: air.channel @channel_{{.*}} [1, 1]
// CHECK-LABEL: func.func @pad_test
// The external channel.put (at segment level) should carry the padding.
// CHECK: air.segment
// CHECK: air.channel.put{{.*}}@channel_
// CHECK-SAME: pad_after = array<i32: 2, 1>
// CHECK-SAME: pad_before = array<i32: 0, 2>
// The internal channel.get (inside herd) should NOT have padding.
// CHECK: air.herd
// CHECK: air.channel.get{{.*}}@channel_

module {
  func.func @pad_test(%arg0: memref<16x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<16x16xi32> {
      air.segment @seg args(%seg_a=%a) : memref<16x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %alloc_l2 = memref.alloc() : memref<16x16xi32, 1>
        air.dma_memcpy_nd (%alloc_l2[] [] [], %seg_a[] [] []) : (memref<16x16xi32, 1>, memref<16x16xi32>)
        air.herd @compute tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c1_s) args(%l2=%alloc_l2) : memref<16x16xi32, 1> {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c13 = arith.constant 13 : index
          %c14 = arith.constant 14 : index
          %alloc_l1 = memref.alloc() : memref<16x16xi32, 2>
          air.dma_memcpy_nd (%alloc_l1[] [] [], %l2[%c0_h, %c0_h] [%c14, %c13] [%c13, %c1_h]) {pad_before = array<i32: 0, 2>, pad_after = array<i32: 2, 1>} : (memref<16x16xi32, 2>, memref<16x16xi32, 1>)
          memref.dealloc %alloc_l1 : memref<16x16xi32, 2>
          air.herd_terminator
        }
        memref.dealloc %alloc_l2 : memref<16x16xi32, 1>
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
