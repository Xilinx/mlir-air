//===- opt_shim_dma_bds_shared_shim_channel.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// Two puts of one buffer on @a, with a put on @b between them. @a and @b
// are queued on the same shim DMA channel, so folding the @a puts into the
// last one would move the first @a transfer behind the @b one.

// CHECK-LABEL: func.func @shared_shim_channel
// CHECK: air.channel.put {{.*}}@a[]
// CHECK: air.channel.put {{.*}}@b[]
// CHECK: air.channel.put {{.*}}@a[]

module {
  aie.device(npu2) {
    %shim = aie.tile(5, 0)
    aie.shim_dma_allocation @air_a(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_b(%shim, MM2S, 0)
  }
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @shared_shim_channel(%x: memref<*xbf16>, %y: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %0 = air.wait_all async
    %p = air.channel.put async [%0] @a[] (%x[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_a", index = 0 : i32}]} : (memref<*xbf16>)
    %q = air.channel.put async [%0] @b[] (%y[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_b", index = 0 : i32}]} : (memref<*xbf16>)
    %r = air.channel.put async [%p] @a[] (%x[%c512] [%c512] [%c1]) {metadataArray = [{base = "air_a", index = 0 : i32}]} : (memref<*xbf16>)
    return
  }
}

// -----

// The same, with @b on a shim channel of its own: the @a puts fold.

// CHECK-LABEL: func.func @own_shim_channel
// CHECK: air.channel.put {{.*}}@b[]
// CHECK: air.channel.put {{.*}}@a[] (%arg0[%c0{{.*}}] [%c1024] [%c1])
// CHECK-NOT: air.channel.put

module {
  aie.device(npu2) {
    %shim = aie.tile(5, 0)
    aie.shim_dma_allocation @air_a(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_b(%shim, MM2S, 1)
  }
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @own_shim_channel(%x: memref<*xbf16>, %y: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %0 = air.wait_all async
    %p = air.channel.put async [%0] @a[] (%x[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_a", index = 0 : i32}]} : (memref<*xbf16>)
    %q = air.channel.put async [%0] @b[] (%y[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_b", index = 0 : i32}]} : (memref<*xbf16>)
    %r = air.channel.put async [%p] @a[] (%x[%c512] [%c512] [%c1]) {metadataArray = [{base = "air_a", index = 0 : i32}]} : (memref<*xbf16>)
    return
  }
}
