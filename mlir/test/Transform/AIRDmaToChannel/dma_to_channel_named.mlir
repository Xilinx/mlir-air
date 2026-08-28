//===- dma_to_channel_named.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A DMA naming a channel lowers onto that declaration instead of a fresh
// @channel_N, and every property the front end wrote on the declaration
// survives untouched.

// RUN: air-opt %s -air-dma-to-channel -split-input-file | FileCheck %s

// CHECK: air.channel @namedChan [] {channel_type = "npu_dma_packet"}
// CHECK-NOT: air.channel @channel_0
// CHECK-LABEL: func.func @named
// CHECK: air.channel.put{{.*}}@namedChan[]
// CHECK: air.herd
// CHECK: air.channel.get{{.*}}@namedChan[]
air.channel @namedChan [] {channel_type = "npu_dma_packet"}
func.func @named(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @namedChan} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// A rank-0 declaration is addressed with no index even though the enclosing
// herd is 2-dimensional. The spatial inference counts herd dimensions and knows
// nothing about the declaration; letting it win would index a rank-0 bundle
// with two indices, which stays malformed all the way to air-to-aie.

// CHECK-LABEL: func.func @named_rank0_under_2d_herd
// CHECK: air.channel.put{{.*}}@flat[]
// CHECK: air.channel.get{{.*}}@flat[]
air.channel @flat []
func.func @named_rank0_under_2d_herd(%arg0: memref<64x64xi32>) {
  %c2 = arith.constant 2 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @flat} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// channel_indices selects a sub-channel of a bundled declaration explicitly.
// It is what you write when the bundle index is NOT the enclosing spatial
// index: the herd here is 2x2, the bundle is [4], and the copy wants lane 2.
// Declared properties survive on the declaration.

// CHECK: air.channel @wFeed [4] {air.shared_resident_ring}
// CHECK-LABEL: func.func @named_explicit_index
// CHECK: air.channel.put{{.*}}@wFeed[%c2]
// CHECK: air.herd
// CHECK: %[[L:.*]] = arith.constant 2 : index
// CHECK: air.channel.get{{.*}}@wFeed[%[[L]]]
air.channel @wFeed [4] {air.shared_resident_ring}
func.func @named_explicit_index(%arg0: memref<64x64xi32>) {
  %c2 = arith.constant 2 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @wFeed, channel_indices = array<i64: 2>} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// An unnamed DMA in the same module still gets a fresh channel, so naming is
// opt-in per op rather than per module.

// CHECK-DAG: air.channel @mixed [
// CHECK-DAG: air.channel @channel_0
// CHECK-LABEL: func.func @named_and_unnamed
air.channel @mixed []
func.func @named_and_unnamed(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    %alloc2 = memref.alloc() : memref<32x32xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @mixed} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    air.dma_memcpy_nd (%alloc2[] [] [], %a[%c32, %c0] [%c32, %c32] [%c64, %cst1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
    memref.dealloc %alloc2 : memref<32x32xi32, 2>
  }
  return
}

// -----

// Several DMAs may name ONE channel: each contributes its own put/get pair, and
// the channel is declared exactly once. This is the convergent multi-producer
// feed that a fresh-channel-per-DMA lowering cannot express.

// CHECK: air.channel @converge [] {channel_type = "npu_dma_packet"}
// CHECK-NOT: air.channel @
// CHECK-LABEL: func.func @named_multi_producer
// CHECK-COUNT-2: air.channel.put{{.*}}@converge[]
// CHECK: air.herd
// CHECK-COUNT-2: air.channel.get{{.*}}@converge[]
air.channel @converge [] {channel_type = "npu_dma_packet"}
func.func @named_multi_producer(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0, %b=%arg1) : memref<64x64xi32>, memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    %alloc2 = memref.alloc() : memref<32x32xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @converge} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    air.dma_memcpy_nd (%alloc2[] [] [], %b[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 2 : i32, channel = @converge} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
    memref.dealloc %alloc2 : memref<32x32xi32, 2>
  }
  return
}
