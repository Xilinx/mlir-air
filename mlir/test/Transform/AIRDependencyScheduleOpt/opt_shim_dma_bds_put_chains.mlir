//===- opt_shim_dma_bds_put_chains.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// Consecutive puts of one L3 buffer on one channel fold into one put, as they
// do for an L2 buffer at the memtile.

// A tile of A and then tiles of B, of another shape, starting where A ends:
// one run.

// CHECK-LABEL: func.func @chain_static
// CHECK: air.channel.put async {{.*}}@ab[] (%arg0[%c0{{.*}}] [16384] [1])
// CHECK-NOT: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @chain_static(%src: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %p = air.channel.put async [%0] @ab[] (%src[%c0, %c0] [%c16, %c512] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    %q = air.channel.put async [%p] @ab[] (%src[%c0, %c0, %c8192] [%c2, %c16, %c256] [%c4096, %c256, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    return
  }
}

// -----

// The same from a start known only at run time.

// CHECK-LABEL: func.func @chain_run_time
// CHECK: air.channel.put async {{.*}}@ab[] (%arg0[%arg1] [16384] [1])
// CHECK-NOT: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @chain_run_time(%src: memref<*xbf16>, %s: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %s2 = arith.addi %s, %c8192 : index
    %p = air.channel.put async [%0] @ab[] (%src[%c0, %s] [%c16, %c512] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    %q = air.channel.put async [%p] @ab[] (%src[%c0, %c0, %s2] [%c2, %c16, %c256] [%c4096, %c256, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    return
  }
}

// -----

// B starts 8 elements after A ends: the puts stay.

// CHECK-LABEL: func.func @chain_gap
// CHECK-COUNT-2: air.channel.put async

module {
  air.channel @ab [1, 1]
  func.func @chain_gap(%src: memref<*xbf16>, %s: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c8192 = arith.constant 8192 : index
    %c8200 = arith.constant 8200 : index
    %0 = air.wait_all async
    %s2 = arith.addi %s, %c8200 : index
    %p = air.channel.put async [%0] @ab[] (%src[%s] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    %q = air.channel.put async [%p] @ab[] (%src[%s2] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    return
  }
}

// -----

// Puts of one shape at evenly spaced offsets: one more dim.

// CHECK-LABEL: func.func @lattice
// CHECK: air.channel.put async {{.*}}@ab[] (%arg0[{{.*}}] [{{.*}}3, {{.*}}64] [{{.*}}256, {{.*}}1])
// CHECK-NOT: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @lattice(%src: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %0 = air.wait_all async
    %p = air.channel.put async [%0] @ab[] (%src[%c0] [%c64] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    %q = air.channel.put async [%p] @ab[] (%src[%c256] [%c64] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    %r = air.channel.put async [%q] @ab[] (%src[%c512] [%c64] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
    return
  }
}
