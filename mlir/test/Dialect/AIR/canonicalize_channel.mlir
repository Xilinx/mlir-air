//===- canonicalize_channel.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize -split-input-file | FileCheck %s

// Prune channels with empty uses.
// CHECK-LABEL: module
// CHECK-NEXT: %c2 = arith.constant 2 : index
// CHECK-NEXT: %c1 = arith.constant 1 : index
// CHECK-NEXT: air.channel @channel_0 [1, 1]
// CHECK-NEXT: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c1, %{{.*}}=%c2) attributes {foo = "bar"} {

module {
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  air.launch (%tx, %ty) in (%size_x = %c1, %size_y = %c2) attributes {foo = "bar"} {
    %alloc = memref.alloc() : memref<16x8xi32>
    air.channel.get  @channel_0[] (%alloc[] [] []) : (memref<16x8xi32>)
  }
}

//
// -----
// CHECK-LABEL: module
// CHECK: func.func @collapse_shape_channel_put_fold
// CHECK-NOT: memref.collapse_shape
// CHECK: air.channel.put @channel_3[] (%alloc[] [] []) : (memref<256x1xf32>)
// CHECK: return
module {
  air.channel @channel_3 [1, 1]
  func.func @collapse_shape_channel_put_fold() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x1xf32>
    %collapsed = memref.collapse_shape %alloc [[0, 1]] : memref<256x1xf32> into memref<256xf32>
    air.channel.put @channel_3[] (%collapsed[%c0] [%c256] [%c1]) : (memref<256xf32>)
    return
  }
}

//
// -----
// CHECK-LABEL: module
// CHECK: func.func @expand_shape_channel_put_fold
// CHECK-NOT: memref.expand_shape
// CHECK: air.channel.put @channel_2[] (%alloc[] [] []) : (memref<256xf32>)
// CHECK: return
module {
  air.channel @channel_2 [1, 1]
  func.func @expand_shape_channel_put_fold() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256xf32>
    %expanded = memref.expand_shape %alloc [[0, 1]] output_shape [256, 1] : memref<256xf32> into memref<256x1xf32>
    air.channel.put @channel_2[] (%expanded[%c0, %c0] [%c256, %c1] [%c1, %c1]) : (memref<256x1xf32>)
    return
  }
}
