//===- dma_to_channel_fuse_into_producer_loop.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// Hoisting a transfer out of a hierarchy clones its whole enclosing loop nest.
// When the buffer it reads is already filled by a loop standing right next to
// the hierarchy -- the shape every staged L3 -> L2 -> L1 feed has -- the clone
// lands as a second loop over the same buffer with the same trip count, because
// nothing tells it the transfer belongs in the loop that is already there.
//
// Two things go wrong, and only the second is visible downstream:
//
//   - The derived put's only incoming dependency is the buffer's ALLOC token.
//     The edge to the get that writes the buffer is simply absent.
//   - air-fuse-alloc-dealloc can then no longer sink the alloc into a loop,
//     since the uses are split across two. air-label-scf-for-to-ping-pong keys
//     on a loop owning its buffer, so it skips it, and a producer that should
//     become a ring of N independently locked slots is emitted as N/2
//     double-buffered pairs -- same buffers, same bytes, coarser
//     synchronisation, six fewer locks per memtile.
//
// So: ONE loop out, with the put inside it, and the put must name the get.

// CHECK-LABEL: func.func @kv
// CHECK: air.segment
// The fill loop is the only segment-level loop left; the derived put joined it.
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: %[[GET:.*]] = air.channel.get async{{.*}}@fill
// CHECK: air.channel.put async [{{.*}}%[[GET]]{{.*}}@drain
// CHECK: scf.yield
// CHECK-NOT: scf.for
// CHECK: air.herd

air.channel @fill [1]
air.channel @drain [1]
func.func @kv(%arg0: memref<4096xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<4096xbf16> {
    air.segment @seg {
      %c0_s = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2_s = arith.constant 2 : index
      %buf = memref.alloc() : memref<4096xbf16, 1>
      scf.for %i = %c0_s to %c2_s step %c1_s {
        air.channel.get @fill[%c0_s] (%buf[] [] []) : (memref<4096xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%b=%buf) : memref<4096xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %is0 = arith.cmpi eq, %tx, %c0_h : index
        scf.if %is0 {
          scf.for %j = %c0_h to %c2_h step %c1_h {
            %l1 = memref.alloc() : memref<2048xbf16, 2>
            air.dma_memcpy_nd (%l1[] [] [], %b[%c0_h] [%c2_h] [%c1_h]) {id = 1 : i32, channel = @drain, channel_indices = array<i64: 0>} : (memref<2048xbf16, 2>, memref<4096xbf16, 1>)
            memref.dealloc %l1 : memref<2048xbf16, 2>
          }
        }
      }
      memref.dealloc %buf : memref<4096xbf16, 1>
    }
  }
  return
}
