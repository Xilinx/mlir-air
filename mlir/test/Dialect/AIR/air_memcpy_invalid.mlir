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
