//===- air_memcpy_invalid.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --split-input-file --verify-diagnostics %s

// -----

// Test: dma_memcpy_nd src sizes/strides rank mismatch.
func.func @dma_src_sizes_strides_mismatch(%m0: memref<64xi32, 2>, %m1: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{'air.dma_memcpy_nd' op src sizes and strides must have the same number of dimensions, but got 2 and 1}}
  air.dma_memcpy_nd (%m0[] [] [], %m1[%c0, %c0] [%c64, %c64] [%c1]) : (memref<64xi32, 2>, memref<64xi32>)
  return
}

// -----

// Test: dma_memcpy_nd dst sizes/strides rank mismatch (src is valid).
func.func @dma_dst_sizes_strides_mismatch(%m0: memref<64xi32, 2>, %m1: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{'air.dma_memcpy_nd' op dst sizes and strides must have the same number of dimensions, but got 1 and 2}}
  air.dma_memcpy_nd (%m0[%c0, %c0] [%c64] [%c1, %c1], %m1[%c0] [%c64] [%c1]) : (memref<64xi32, 2>, memref<64xi32>)
  return
}

// -----

// Test: channel.put src sizes/strides rank mismatch.
air.channel @channel_put_test [1, 1]
func.func @channel_put_src_mismatch(%m: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{'air.channel.put' op src sizes and strides must have the same number of dimensions, but got 1 and 2}}
  air.channel.put @channel_put_test[] (%m[%c0, %c0] [%c64] [%c1, %c1]) : (memref<64xi32>)
  return
}

// -----

// Test: channel.get dst sizes/strides rank mismatch.
air.channel @channel_get_test [1, 1]
func.func @channel_get_dst_mismatch(%m: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{'air.channel.get' op dst sizes and strides must have the same number of dimensions, but got 2 and 1}}
  air.channel.get @channel_get_test[] (%m[%c0] [%c64, %c64] [%c1]) : (memref<64xi32>)
  return
}

// -----

// Test: src_rank requires an enclosing air.rank scope.
func.func @dma_src_rank_no_enclosing_rank(%dst: memref<128xf32, 2>, %src: memref<128xf32>) {
  // expected-error @+1 {{'air.dma_memcpy_nd' op src_rank/dst_rank attributes require an enclosing air.rank scope}}
  air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {src_rank = 0 : i64}
      : (memref<128xf32, 2>, memref<128xf32>)
  return
}

// -----

// Test: dst_rank requires an enclosing air.rank scope.
func.func @dma_dst_rank_no_enclosing_rank(%dst: memref<128xf32>, %src: memref<128xf32, 2>) {
  // expected-error @+1 {{'air.dma_memcpy_nd' op src_rank/dst_rank attributes require an enclosing air.rank scope}}
  air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {dst_rank = 1 : i64}
      : (memref<128xf32>, memref<128xf32, 2>)
  return
}

// -----

// Test: src_rank requires the source memref.alloc to carry the air.symmetric attribute.
func.func @dma_src_rank_alloc_not_symmetric() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %src = memref.alloc() : memref<128xf32>
    %dst = memref.alloc() : memref<128xf32, 2>
    // expected-error @+1 {{'air.dma_memcpy_nd' op src memref is referenced cross-rank but its memref.alloc lacks the "air.symmetric" attribute}}
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {src_rank = 0 : i64}
        : (memref<128xf32, 2>, memref<128xf32>)
  }
  return
}

// -----

// Test: dst_rank requires the destination memref.alloc to carry the air.symmetric attribute.
func.func @dma_dst_rank_alloc_not_symmetric() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %dst = memref.alloc() : memref<128xf32>
    %src = memref.alloc() : memref<128xf32, 2>
    // expected-error @+1 {{'air.dma_memcpy_nd' op dst memref is referenced cross-rank but its memref.alloc lacks the "air.symmetric" attribute}}
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {dst_rank = 1 : i64}
        : (memref<128xf32>, memref<128xf32, 2>)
  }
  return
}

// -----

// Test: src_rank must be non-negative.
func.func @dma_src_rank_negative() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %dst = memref.alloc() : memref<128xf32, 2>
    %src = memref.alloc() {air.symmetric} : memref<128xf32>
    // expected-error @+1 {{'air.dma_memcpy_nd' op src_rank must be >= 0, got -1}}
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {src_rank = -1 : i64}
        : (memref<128xf32, 2>, memref<128xf32>)
  }
  return
}

// -----

// Test: dst_rank must be non-negative.
func.func @dma_dst_rank_negative() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %dst = memref.alloc() {air.symmetric} : memref<128xf32>
    %src = memref.alloc() : memref<128xf32, 2>
    // expected-error @+1 {{'air.dma_memcpy_nd' op dst_rank must be >= 0, got -3}}
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {dst_rank = -3 : i64}
        : (memref<128xf32>, memref<128xf32, 2>)
  }
  return
}
