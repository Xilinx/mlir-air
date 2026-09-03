//===- dma_to_channel_fifo_interleave.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// Several transfers on ONE channel sharing ONE loop must come out of the hoist
// sharing one loop.
//
// The driver marks one transfer "external" and runs the hoisting patterns, so
// each transfer is hoisted in its own round and each round rebuilds the
// enclosing loop for itself. Two rebuilt loops in sequence do not mean what one
// loop containing two transfers meant: the loop INTERLEAVES them (a, b, a, b)
// and the pair of loops SERIALISES them (a*N, then b*N).
//
// A channel is a FIFO. Its consumer's Nth transfer takes the Nth arrival, so
// serialising one side sends arrival 1 where arrival 2 belonged and the data
// lands at the wrong offsets. It is not a scheduling difference: the same shape
// in fused_decode's GLU output feed produced wrong tokens on device.
//
// The producer side below is the one to read: two puts inside one scf.for, on
// one channel, at alternating 512-word slots of an 8192-word L2 buffer -- a
// ping/pong compute ring writing into a memtile. The derived consumer has to be
// two gets in one loop, not two loops of one get.

// CHECK-LABEL: func.func @two_on_one
// Both derived gets in ONE loop, in program order.
// CHECK: air.segment
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@c
// No second loop between them: that is the whole assertion.
// CHECK-NOT: scf.for
// CHECK: air.channel.get{{.*}}@c
// CHECK-NOT: air.channel.get{{.*}}@c
// CHECK: air.herd

air.channel @c [1]
func.func @two_on_one(%arg0: memref<8192xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<8192xbf16> {
    air.segment @seg args(%sa=%la) : memref<8192xbf16> {
      %c1_s = arith.constant 1 : index
      %l2 = memref.alloc() : memref<8192xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%d=%l2) : memref<8192xbf16, 1> {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        scf.for %i = %c0 to %c8 step %c1_h {
          %o0 = arith.muli %i, %c1024 : index
          %a = memref.alloc() : memref<512xbf16, 2>
          air.dma_memcpy_nd (%d[%o0] [%c512] [%c1_h], %a[] [] []) {id = 1 : i32, channel = @c, channel_indices = array<i64: 0>} : (memref<8192xbf16, 1>, memref<512xbf16, 2>)
          memref.dealloc %a : memref<512xbf16, 2>
          %o1 = arith.addi %o0, %c512 : index
          %b = memref.alloc() : memref<512xbf16, 2>
          air.dma_memcpy_nd (%d[%o1] [%c512] [%c1_h], %b[] [] []) {id = 2 : i32, channel = @c, channel_indices = array<i64: 0>} : (memref<8192xbf16, 1>, memref<512xbf16, 2>)
          memref.dealloc %b : memref<512xbf16, 2>
        }
      }
      memref.dealloc %l2 : memref<8192xbf16, 1>
    }
  }
  return
}
