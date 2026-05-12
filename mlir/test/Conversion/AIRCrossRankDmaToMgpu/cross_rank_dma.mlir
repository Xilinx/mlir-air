//===- cross_rank_dma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// RUN: air-opt %s --split-input-file -air-cross-rank-dma-to-mgpu | FileCheck %s

// Each test wraps the cross-rank dma in air.rank to satisfy the verifier
// (added in Phase 1) that requires an enclosing air.rank scope.

// Basic src_rank: lower to mgpuMemcpy with peer-VA addressing.
// CHECK-LABEL: func.func @src_rank
// CHECK: arith.constant 4096 : i64
// CHECK: llvm.mlir.zero : !llvm.ptr
// Extract pointers from both memrefs.
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: memref.extract_aligned_pointer_as_index
// Get bases and rank.
// CHECK: call @mgpuGetHeapBases() : () -> !llvm.ptr
// CHECK: call @mgpuGetRank() : () -> i32
// CHECK: arith.extsi
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// peer rank constant (0).
// CHECK: llvm.mlir.constant(0 : i64)
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// offset = peer_local_int - my_base_int.
// CHECK: llvm.ptrtoint
// CHECK: llvm.ptrtoint
// CHECK: arith.subi
// peer_ptr = peer_base + offset (byte stride GEP).
// CHECK: llvm.getelementptr {{.*}} -> !llvm.ptr, i8
// Final memcpy call.
// CHECK: call @mgpuMemcpy
// CHECK-NOT: air.dma_memcpy_nd
func.func @src_rank(%dst: memref<1024xf32>, %src: memref<1024xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%d = %dst, %s = %src)
      : memref<1024xf32>, memref<1024xf32> {
    air.dma_memcpy_nd (%d[] [] [], %s[] [] []) {src_rank = 0 : i64}
        : (memref<1024xf32>, memref<1024xf32>)
    air.rank_terminator
  }
  return
}

// -----

// dst_rank: same lowering pattern, peer pointer becomes the dst arg.
// CHECK-LABEL: func.func @dst_rank
// CHECK: call @mgpuMemcpy
// CHECK-NOT: air.dma_memcpy_nd
func.func @dst_rank(%dst: memref<1024xf32>, %src: memref<1024xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%d = %dst, %s = %src)
      : memref<1024xf32>, memref<1024xf32> {
    air.dma_memcpy_nd (%d[] [] [], %s[] [] []) {dst_rank = 1 : i64}
        : (memref<1024xf32>, memref<1024xf32>)
    air.rank_terminator
  }
  return
}

// -----

// 2D memref byte size: 64 * 64 * 4 = 16384.
// CHECK-LABEL: func.func @cross_rank_2d
// CHECK: arith.constant 16384 : i64
// CHECK: call @mgpuMemcpy
func.func @cross_rank_2d(%dst: memref<64x64xf32>, %src: memref<64x64xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%d = %dst, %s = %src)
      : memref<64x64xf32>, memref<64x64xf32> {
    air.dma_memcpy_nd (%d[] [] [], %s[] [] []) {src_rank = 0 : i64}
        : (memref<64x64xf32>, memref<64x64xf32>)
    air.rank_terminator
  }
  return
}

// -----

// f64 element type: 256 * 8 = 2048 bytes.
// CHECK-LABEL: func.func @cross_rank_f64
// CHECK: arith.constant 2048 : i64
func.func @cross_rank_f64(%dst: memref<256xf64>, %src: memref<256xf64>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%d = %dst, %s = %src)
      : memref<256xf64>, memref<256xf64> {
    air.dma_memcpy_nd (%d[] [] [], %s[] [] []) {src_rank = 0 : i64}
        : (memref<256xf64>, memref<256xf64>)
    air.rank_terminator
  }
  return
}

// -----

// Multiple cross-rank DMAs in one function: extern decls emitted exactly once.
// Match emission order from ensureExternFunc (insertion-at-top -> reverse).
// CHECK-COUNT-1: func.func private @mgpuMemcpy
// CHECK-NOT: func.func private @mgpuMemcpy
// CHECK-COUNT-1: func.func private @mgpuGetHeapBases
// CHECK-NOT: func.func private @mgpuGetHeapBases
// CHECK-COUNT-1: func.func private @mgpuGetRank
// CHECK-NOT: func.func private @mgpuGetRank
// CHECK-LABEL: func.func @two_dmas
// CHECK-COUNT-2: call @mgpuMemcpy
func.func @two_dmas(%dst: memref<32xf32>, %src: memref<32xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%d = %dst, %s = %src)
      : memref<32xf32>, memref<32xf32> {
    air.dma_memcpy_nd (%d[] [] [], %s[] [] []) {src_rank = 0 : i64}
        : (memref<32xf32>, memref<32xf32>)
    air.dma_memcpy_nd (%d[] [] [], %s[] [] []) {src_rank = 0 : i64}
        : (memref<32xf32>, memref<32xf32>)
    air.rank_terminator
  }
  return
}

// -----

// LAST partition: pass is a no-op for non-cross-rank DMAs.
// CHECK-LABEL: func.func @no_cross_rank
// CHECK: air.dma_memcpy_nd
// CHECK-NOT: mgpuMemcpy
// CHECK-NOT: mgpuGetHeapBases
func.func @no_cross_rank(%dst: memref<1024xf32, 2>, %src: memref<1024xf32>) {
  air.dma_memcpy_nd (%dst[] [] [], %src[] [] [])
      : (memref<1024xf32, 2>, memref<1024xf32>)
  return
}
