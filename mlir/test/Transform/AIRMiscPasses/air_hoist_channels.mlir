//===- air_hoist_channels.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-hoist-channels %s | FileCheck %s

// CHECK: put @channel_0[%c0, %c0] (%arg0[%c0] [%c4, %c32] [%c32, %c1]) : (memref<128xf32>)
// CHECK: get @channel_1[%c0, %c0] (%arg1[%c0, %c0] [%c4, %c32, %c32] [%c4096, %c128, %c1]) : (memref<128x128xf32>)
// CHECK: put @channel_2[%c0, %c0] (%arg1[%c0, %c0] [%c4, %c32, %c32] [%c32, %c32, %c1]) : (memref<128x128xf32>)
#map = affine_map<(d0) -> (d0)>
module {
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @t0(%arg0: memref<128xf32>, %arg1: memref<128x128xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<128xf32>
    scf.for %arg2 = %c0 to %c128 step %c32 {
      air.channel.put  @channel_0[%c0, %c0] (%arg0[%arg2] [%c32] [%c1]) : (memref<128xf32>)
    }
    scf.for %arg2 = %c0 to %c128 step %c32 {
      air.channel.get  @channel_1[%c0, %c0] (%arg1[%arg2, %c0] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
    }
    scf.for %arg2 = %c0 to %c128 step %c32 {
      air.channel.put  @channel_2[%c0, %c0] (%arg1[%c0, %arg2] [%c32, %c32] [%c32, %c1]) : (memref<128x128xf32>)
    }
    return %alloc : memref<128xf32>
  }
}
