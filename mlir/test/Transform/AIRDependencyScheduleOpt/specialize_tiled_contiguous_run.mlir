//===- specialize_tiled_contiguous_run.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-opt-memtile-dma-bds='device=npu2' | FileCheck %s

// Several channel ops in one loop body that TILE a contiguous run are one
// transfer written N ways, and have to fold to one descriptor.
//
// The loop fold was gated on the body holding EXACTLY one channel op. With one,
// `scf.for %i to 16 { get(b[%i*512], 512) }` already becomes a single
// whole-buffer transfer -- the minimal legal descriptor. With two ops covering
// the identical region (a ping/pong producer, writing base and base+size each
// iteration) the gate declined, AIRUnrollScfForIntoBDChain unrolled the nest,
// and the same 8192 words came out as SIXTEEN descriptors at offsets
// 0, 512 ... 7680. Both are legal; only one is the answer the single-op case
// gives.
//
// The difference is not cosmetic. On a memtile fill each descriptor is its own
// lock release, so a consumer whose counting lock was derived from the
// single-descriptor form -- an `air.refeed_count` re-read of the resident
// buffer, say -- waits for N and is released N*16 times. fused_decode's GLU
// output feed is exactly that shape and timed out on device at decode pos0
// until this folded.
//
// Only a genuine tiling folds: same channel, same direction, same memref and
// indices, 1-D, unit stride, equal sizes, and offset by exactly one size.
// Anything else still declines and unrolls as before.

// CHECK-LABEL: func.func @tiled_run
// One transfer over the whole buffer, and no leftover loop.
// CHECK: air.segment
// CHECK: air.channel.get{{.*}}@c[%c0] (%results[] [] [])
// CHECK-NOT: air.channel.get{{.*}}@c
// CHECK-NOT: scf.for

air.channel @c [1]
func.func @tiled_run() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c1024 = arith.constant 1024 : index
      %b = memref.alloc() : memref<8192xbf16, 1>
      // Ping/pong: two adjacent 512-word slots per iteration. Their union is
      // the loop's own offset step, so the nest is one linear 8192 run.
      scf.for %i = %c0 to %c8 step %c1_s {
        %o = arith.muli %i, %c1024 : index
        %o2 = arith.addi %o, %c512 : index
        air.channel.get @c[%c0] (%b[%o] [%c512] [%c1_s]) : (memref<8192xbf16, 1>)
        air.channel.get @c[%c0] (%b[%o2] [%c512] [%c1_s]) : (memref<8192xbf16, 1>)
      }
      memref.dealloc %b : memref<8192xbf16, 1>
    }
  }
  return
}
