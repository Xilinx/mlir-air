//===- air_cross_rank_dma.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Round-trip tests for air.dma_memcpy_nd with src_rank/dst_rank attributes
// and for memref.alloc with the air.symmetric attribute. The cross-rank
// attributes require an enclosing air.rank scope.
//
// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_dma_with_src_rank
func.func @test_dma_with_src_rank() {
  %c2 = arith.constant 2 : index
  // CHECK: air.rank
  air.rank (%rx) in (%sx = %c2) {
    // CHECK: %[[BUF:.*]] = memref.alloc() {air.symmetric} : memref<128xf32>
    %buf = memref.alloc() {air.symmetric} : memref<128xf32>
    %local = memref.alloc() : memref<128xf32, 2>
    // CHECK: air.dma_memcpy_nd
    // CHECK-SAME: src_rank = 0
    air.dma_memcpy_nd (%local[] [] [], %buf[] [] []) {src_rank = 0 : i64}
        : (memref<128xf32, 2>, memref<128xf32>)
  }
  return
}

// CHECK-LABEL: func.func @test_dma_with_dst_rank
func.func @test_dma_with_dst_rank() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %buf = memref.alloc() {air.symmetric} : memref<128xf32>
    %local = memref.alloc() : memref<128xf32, 2>
    // CHECK: air.dma_memcpy_nd
    // CHECK-SAME: dst_rank = 1
    air.dma_memcpy_nd (%buf[] [] [], %local[] [] []) {dst_rank = 1 : i64}
        : (memref<128xf32>, memref<128xf32, 2>)
  }
  return
}

// CHECK-LABEL: func.func @test_dma_with_both_ranks
func.func @test_dma_with_both_ranks() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %src = memref.alloc() {air.symmetric} : memref<128xf32>
    %dst = memref.alloc() {air.symmetric} : memref<128xf32>
    // CHECK: air.dma_memcpy_nd
    // CHECK-SAME: dst_rank = 1
    // CHECK-SAME: src_rank = 0
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] [])
        {src_rank = 0 : i64, dst_rank = 1 : i64}
        : (memref<128xf32>, memref<128xf32>)
  }
  return
}

// CHECK: air.channel @sym_chan
// CHECK-SAME: channel_type = "gpu_symmetric_heap"
air.channel @sym_chan [] {channel_type = "gpu_symmetric_heap"}

// CHECK-LABEL: func.func @test_sym_channel_put_get_in_rank
func.func @test_sym_channel_put_get_in_rank() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %buf = memref.alloc() : memref<128xf32>
    // CHECK: air.channel.put @sym_chan
    air.channel.put @sym_chan[] (%buf[] [] []) : (memref<128xf32>)
    // CHECK: air.channel.get @sym_chan
    air.channel.get @sym_chan[] (%buf[] [] []) : (memref<128xf32>)
  }
  return
}
