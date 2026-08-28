//===- dma_to_channel_refeed.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// End to end in aircc's pass order: a re-feed loop written over a DMA is
// collapsed by air-annotate-refeed (front of the placement pipeline), and
// air-dma-to-channel carries the count onto the put it generates -- the op
// air-to-aie's allocateCoreLocksPerMemcpyOp reads to scale the core-side
// release. The result is the same IR the front end would have written by hand
// as an n-trip loop around an air.channel.put.

// RUN: air-opt %s -air-annotate-refeed -air-dependency -air-dma-to-channel | FileCheck %s

// CHECK-LABEL: func.func @refeed_reaches_the_put
// CHECK: air.channel.put
// CHECK-SAME: @xfeed
// CHECK-SAME: air.refeed_count = 6 : i32
// The consuming half must NOT carry the count: it is the producer's release
// that is scaled.
// CHECK: air.channel.get
// CHECK-NOT: air.refeed_count
air.channel @xfeed []
func.func @refeed_reaches_the_put(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %cst1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    scf.for %i = %c0 to %c6 step %cst1 {
      air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @xfeed} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    }
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}
