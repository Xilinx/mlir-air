//===- gpu_channel.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// RUN: air-opt %s --split-input-file -air-gpu-channel-to-mgpu | FileCheck %s

// Basic put/get pair with peer-rank index. The put becomes a barrier; the
// get becomes barrier + cross-rank mgpuMemcpy.
// CHECK-LABEL: func.func @basic_pair
// CHECK-NOT: air.channel @
// CHECK: arith.constant 0 : index
// Inside the rank body: put -> barrier
// CHECK: call @mgpuBarrier
// CHECK-NOT: air.channel.put
// Then: get -> barrier + memcpy with peer-VA addressing.
// CHECK: call @mgpuBarrier
// CHECK: arith.constant 4096 : i64
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: call @mgpuGetHeapBases
// CHECK: call @mgpuGetRank
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// peer rank = constant 0 (peer index from get).
// CHECK: arith.index_cast
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// offset = src_int - my_base_int.
// CHECK: llvm.ptrtoint
// CHECK: llvm.ptrtoint
// CHECK: arith.subi
// peer_src = peer_base + offset (byte stride).
// CHECK: llvm.getelementptr {{.*}} -> !llvm.ptr, i8
// CHECK: call @mgpuMemcpy
// CHECK-NOT: air.channel.get
air.channel @sym_chan [] {channel_type = "gpu_symmetric_heap"}
func.func @basic_pair(%src: memref<1024xf32>, %dst: memref<1024xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%s = %src, %d = %dst)
      : memref<1024xf32>, memref<1024xf32> {
    %c0 = arith.constant 0 : index
    %sym = memref.alloc() {air.symmetric} : memref<1024xf32>
    air.channel.put @sym_chan[] (%sym[] [] []) : (memref<1024xf32>)
    air.channel.get @sym_chan[%c0] (%d[] [] []) : (memref<1024xf32>)
    memref.dealloc %sym : memref<1024xf32>
    air.rank_terminator
  }
  return
}

// -----

// Channel decl is erased after lowering (the channel symbol no longer
// exists in the lowered IR).
// CHECK-LABEL: func.func @decl_erased
// CHECK-NOT: air.channel @sym_chan2
air.channel @sym_chan2 [] {channel_type = "gpu_symmetric_heap"}
func.func @decl_erased(%dst: memref<32xf32>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%d = %dst)
      : memref<32xf32> {
    %c0 = arith.constant 0 : index
    %sym = memref.alloc() {air.symmetric} : memref<32xf32>
    air.channel.put @sym_chan2[] (%sym[] [] []) : (memref<32xf32>)
    air.channel.get @sym_chan2[%c0] (%d[] [] []) : (memref<32xf32>)
    memref.dealloc %sym : memref<32xf32>
    air.rank_terminator
  }
  return
}

// -----

// LAST partition: pass is a no-op for non-gpu_symmetric_heap channels.
// (npu_dma_stream channels must be left alone for the AIE backend.)
// CHECK-LABEL: func.func @no_gpu_channel
// CHECK: air.channel.put @npu_chan
// CHECK-NOT: mgpuMemcpy
// CHECK-NOT: mgpuGetHeapBases
air.channel @npu_chan [] {channel_type = "npu_dma_stream"}
func.func @no_gpu_channel(%src: memref<32xf32>) {
  air.channel.put @npu_chan[] (%src[] [] []) : (memref<32xf32>)
  return
}
