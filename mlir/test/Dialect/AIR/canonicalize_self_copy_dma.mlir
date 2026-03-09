//===- canonicalize_self_copy_dma.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize %s | FileCheck %s

// Self-copy DMA (non-async) should be erased.
// CHECK-LABEL: func.func @self_copy_dma
// CHECK-NOT: air.dma_memcpy_nd
// CHECK: return
func.func @self_copy_dma(%arg0: memref<16x16xf32, 2>) {
  air.dma_memcpy_nd (%arg0[] [] [], %arg0[] [] []) {id = 1 : i32} : (memref<16x16xf32, 2>, memref<16x16xf32, 2>)
  return
}

// Self-copy DMA (async) should be replaced with wait_all to preserve the
// dependency chain. In this test, the wait_all result is only used by another
// wait_all with no users, so canonicalization removes everything.
// CHECK-LABEL: func.func @self_copy_dma_async
// CHECK-NOT: air.dma_memcpy_nd
// CHECK: return
func.func @self_copy_dma_async(%arg0: memref<16x16xf32, 2>) {
  %0 = air.dma_memcpy_nd async (%arg0[] [] [], %arg0[] [] []) {id = 1 : i32} : (memref<16x16xf32, 2>, memref<16x16xf32, 2>)
  air.wait_all [%0]
  return
}

// Self-copy DMA (async) where the token is used: the self-copy DMA is replaced
// with a wait_all, and then canonicalization removes the empty wait_all,
// leaving the dependent DMA without the stale dependency.
// CHECK-LABEL: func.func @self_copy_dma_async_used
// CHECK-NOT: air.dma_memcpy_nd{{.*}}(%arg0[] [] [], %arg0[] [] [])
// CHECK: air.dma_memcpy_nd async (%arg0[] [] [], %arg1[] [] [])
func.func @self_copy_dma_async_used(%arg0: memref<16x16xf32, 2>, %arg1: memref<16x16xf32>) {
  %0 = air.dma_memcpy_nd async (%arg0[] [] [], %arg0[] [] []) {id = 1 : i32} : (memref<16x16xf32, 2>, memref<16x16xf32, 2>)
  %1 = air.dma_memcpy_nd async [%0] (%arg0[] [] [], %arg1[] [] []) {id = 2 : i32} : (memref<16x16xf32, 2>, memref<16x16xf32>)
  air.wait_all [%1]
  return
}

// Self-copy DMA with offsets/sizes/strides should be erased when they match.
// CHECK-LABEL: func.func @self_copy_dma_with_offsets
// CHECK-NOT: air.dma_memcpy_nd
// CHECK: return
func.func @self_copy_dma_with_offsets(%arg0: memref<64xbf16, 1>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  air.dma_memcpy_nd (%arg0[%c0] [%c32] [%c1], %arg0[%c0] [%c32] [%c1]) {id = 1 : i32} : (memref<64xbf16, 1>, memref<64xbf16, 1>)
  return
}

// Different src and dst should NOT be erased.
// CHECK-LABEL: func.func @different_src_dst_dma
// CHECK: air.dma_memcpy_nd
func.func @different_src_dst_dma(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32, 2>) {
  air.dma_memcpy_nd (%arg1[] [] [], %arg0[] [] []) {id = 1 : i32} : (memref<16x16xf32, 2>, memref<16x16xf32>)
  return
}
