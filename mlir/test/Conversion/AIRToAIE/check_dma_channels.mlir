//===- check_dma_channels.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-aie -verify-diagnostics %s

// expected-error@-10 {{'AIE.device' op : lowering of segments containing both dma copies and channels is not supported}}
air.channel @channel_0 [1, 1]
func.func @test(%arg0: memref<1024xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @bad_herd  tile (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<1024xi32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %arg5[%c0] [%c0_1] [%c0_0]) : (memref<1024xi32, 2>, memref<1024xi32>)
    air.channel.get  @channel_0[] (%alloc[%c0] [%c0_1] [%c0_0]) : (memref<1024xi32, 2>)
    memref.dealloc %alloc : memref<1024xi32, 2>
    air.herd_terminator
  }
  return
}
