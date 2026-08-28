//===- air_annotate_refeed_dma.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-annotate-refeed runs at the FRONT of the placement pipeline, before
// air-dma-to-channel. A re-feed written over an air.dma_memcpy_nd therefore has
// to be recognized in its DMA form or it is never recognized at all -- and
// air-opt-memtile-dma-bds would go on to erase the loop as a redundant
// single-BD wrapper, silently dropping the re-sends.

// RUN: air-opt %s -air-annotate-refeed -split-input-file | FileCheck %s

// An L3 source: the count rides the transfer itself.
// CHECK-LABEL: func.func @refeed_dma_l3
// CHECK-NOT: scf.for
// CHECK: air.dma_memcpy_nd
// CHECK-SAME: air.refeed_count = 6 : i32
air.channel @xfeed []
func.func @refeed_dma_l3(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %cst1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    scf.for %i = %c0 to %c6 step %cst1 {
      air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @xfeed} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    }
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// An L2 source: the count goes on the backing memref.alloc instead, because the
// memtile fill/drain rendezvous is primed from the buffer, not from the
// transfer. Writing it on the transfer would be silently ineffective.
// CHECK-LABEL: func.func @refeed_dma_l2
// CHECK-NOT: scf.for
// CHECK: memref.alloc()
// CHECK-SAME: air.refeed_count = 4 : i32
air.channel @l2feed []
func.func @refeed_dma_l2() {
  %c1 = arith.constant 1 : index
  air.segment {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %l2 = memref.alloc() : memref<32x32xi32, 1>
    %l1 = memref.alloc() : memref<32x32xi32, 2>
    scf.for %i = %c0 to %c4 step %cst1 {
      air.dma_memcpy_nd (%l1[] [] [], %l2[%c0, %c0] [%c32, %c32] [%c32, %cst1]) {id = 1 : i32, channel = @l2feed} : (memref<32x32xi32, 2>, memref<32x32xi32, 1>)
    }
    memref.dealloc %l1 : memref<32x32xi32, 2>
    memref.dealloc %l2 : memref<32x32xi32, 1>
  }
  return
}

// -----

// A trip count of 1 is not a re-feed; the loop is still collapsed but no count
// is recorded (getRefeedCount's absent case already means "1").
// CHECK-LABEL: func.func @refeed_dma_trip1
// CHECK-NOT: air.refeed_count
air.channel @once []
func.func @refeed_dma_trip1(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %cst1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    scf.for %i = %c0 to %cst1 step %cst1 {
      air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @once} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    }
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// A DMA whose source DOES vary with the induction variable is N productions,
// not N re-sends of one buffer. Left alone.
// CHECK-LABEL: func.func @not_a_refeed
// CHECK: scf.for
// CHECK-NOT: air.refeed_count
air.channel @varying []
func.func @not_a_refeed(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %cst1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    scf.for %i = %c0 to %c6 step %cst1 {
      air.dma_memcpy_nd (%alloc[] [] [], %a[%i, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @varying} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    }
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}
