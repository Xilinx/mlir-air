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

// -----

// @a's bundle entry [1, 0] is queued on MM2S 1, as @b is; its entry [0, 0] is
// on MM2S 0. The @a puts at [1, 0] do not fold across the @b put.

// CHECK-LABEL: func.func @bundle_entry
// CHECK: air.channel.put {{.*}}@a[%c1{{.*}}, %c0{{.*}}]
// CHECK: air.channel.put {{.*}}@b[]
// CHECK: air.channel.put {{.*}}@a[%c1{{.*}}, %c0{{.*}}]

module {
  aie.device(npu2) {
    %shim = aie.tile(5, 0)
    aie.shim_dma_allocation @air_a0(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_a1(%shim, MM2S, 1)
    aie.shim_dma_allocation @air_b(%shim, MM2S, 1)
  }
  air.channel @a [2, 1]
  air.channel @b [1, 1]
  func.func @bundle_entry(%x: memref<*xbf16>, %y: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %0 = air.wait_all async
    %p = air.channel.put async [%0] @a[%c1, %c0] (%x[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_a0", index = 0 : i32}, {base = "air_a1", index = 1 : i32}]} : (memref<*xbf16>)
    %q = air.channel.put async [%0] @b[] (%y[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_b", index = 0 : i32}]} : (memref<*xbf16>)
    %r = air.channel.put async [%p] @a[%c1, %c0] (%x[%c512] [%c512] [%c1]) {metadataArray = [{base = "air_a0", index = 0 : i32}, {base = "air_a1", index = 1 : i32}]} : (memref<*xbf16>)
    return
  }
}

// -----

// Before placement: @a and @b share MM2S 0 of a shim tile placed at a
// column hint. The @a puts do not fold.

// CHECK-LABEL: func.func @logical_shim
// CHECK: air.channel.put {{.*}}@a[]
// CHECK: air.channel.put {{.*}}@b[]
// CHECK: air.channel.put {{.*}}@a[]

module {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(5, ?)
    aie.shim_dma_allocation @air_a(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_b(%shim, MM2S, 0)
  }
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @logical_shim(%x: memref<*xbf16>, %y: memref<*xbf16>) {
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

// With no column to place the shim tile at, the channels cannot be told
// apart, so the @a puts keep their order around @b.

// CHECK-LABEL: func.func @unplaced_shim
// CHECK: air.channel.put {{.*}}@a[]
// CHECK: air.channel.put {{.*}}@b[]
// CHECK: air.channel.put {{.*}}@a[]

module {
  aie.device(npu2) {
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.shim_dma_allocation @air_a(%shim_a, MM2S, 0)
    aie.shim_dma_allocation @air_b(%shim_b, MM2S, 0)
  }
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @unplaced_shim(%x: memref<*xbf16>, %y: memref<*xbf16>) {
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

// Two entries of one bundle on the same shim DMA channel: the @a[0, 0] puts
// do not fold across the @a[1, 0] put.

// CHECK-LABEL: func.func @bundle_entries_share
// CHECK: air.channel.put {{.*}}@a[%c0{{.*}}, %c0{{.*}}]
// CHECK: air.channel.put {{.*}}@a[%c1{{.*}}, %c0{{.*}}]
// CHECK: air.channel.put {{.*}}@a[%c0{{.*}}, %c0{{.*}}]

module {
  aie.device(npu2) {
    %shim = aie.tile(5, 0)
    aie.shim_dma_allocation @air_a0(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_a1(%shim, MM2S, 0)
  }
  air.channel @a [2, 1]
  func.func @bundle_entries_share(%x: memref<*xbf16>, %y: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %0 = air.wait_all async
    %p = air.channel.put async [%0] @a[%c0, %c0] (%x[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_a0", index = 0 : i32}, {base = "air_a1", index = 1 : i32}]} : (memref<*xbf16>)
    %q = air.channel.put async [%0] @a[%c1, %c0] (%y[%c0] [%c512] [%c1]) {metadataArray = [{base = "air_a0", index = 0 : i32}, {base = "air_a1", index = 1 : i32}]} : (memref<*xbf16>)
    %r = air.channel.put async [%p] @a[%c0, %c0] (%x[%c512] [%c512] [%c1]) {metadataArray = [{base = "air_a0", index = 0 : i32}, {base = "air_a1", index = 1 : i32}]} : (memref<*xbf16>)
    return
  }
}

// -----

// Two devices name @air_b differently: in one it shares @air_a's channel. The
// @a puts do not fold across the @b put.

// CHECK-LABEL: func.func @symbol_in_two_devices
// CHECK: air.channel.put {{.*}}@a[]
// CHECK: air.channel.put {{.*}}@b[]
// CHECK: air.channel.put {{.*}}@a[]

module {
  aie.device(npu2) @d0 {
    %shim = aie.tile(5, 0)
    aie.shim_dma_allocation @air_a(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_b(%shim, MM2S, 1)
  }
  aie.device(npu2) @d1 {
    %shim = aie.tile(5, 0)
    aie.shim_dma_allocation @air_a(%shim, MM2S, 0)
    aie.shim_dma_allocation @air_b(%shim, MM2S, 0)
  }
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @symbol_in_two_devices(%x: memref<*xbf16>, %y: memref<*xbf16>) {
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
