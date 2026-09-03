//===- specialize_tiled_contiguous_run.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-opt-memtile-dma-bds='device=npu2' -split-input-file | FileCheck %s

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

// -----

// The same run with NO LOOP AROUND IT: N ops side by side. That is what
// air-dma-to-channel derives when it reads the tiling off the buffer sizes
// rather than a loop, and what an unrolled front end emits.
//
// It has to reach the same descriptor the loop form reaches. When it does not,
// the loop form folds to one whole-buffer BD and the sibling form stays N, and
// on a memtile that is N lock releases where the consumer's counting lock
// expects one -- so the design hangs in the sibling spelling only, for no
// reason visible in the source. fused_decode's @inX lost its memtile producer
// BD outright this way: air-to-aie could not build the broadcast flow from two
// half-buffer puts, emitted nothing, and every consumer waited forever.

// CHECK-LABEL: func.func @sibling_run
// One transfer over the whole buffer, the same answer the loop form gives.
// CHECK: air.channel.put{{.*}}@s[%c0] (%results[] [] [])
// CHECK-NOT: air.channel.put{{.*}}@s
air.channel @s [1]
func.func @sibling_run() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %b = memref.alloc() : memref<512xbf16, 1>
      air.channel.put @s[%c0] (%b[0] [256] [1]) : (memref<512xbf16, 1>)
      air.channel.put @s[%c0] (%b[256] [256] [1]) : (memref<512xbf16, 1>)
      memref.dealloc %b : memref<512xbf16, 1>
    }
  }
  return
}

// -----

// NEGATIVE CONTROL, and the one that matters: siblings that do NOT tile a
// contiguous run. The second starts at 384, not 256, so the two describe an
// overlapping region with a hole after it -- there is no single descriptor
// with the same byte sequence. Folding anyway would send 512 words from offset
// 0, silently reading 128 words nobody asked for and skipping the tail.

// CHECK-LABEL: func.func @sibling_not_contiguous
// CHECK: air.channel.put{{.*}}@n2[%c0] (%results[0] [256] [1])
// CHECK: air.channel.put{{.*}}@n2[%c0] (%results[384] [256] [1])
air.channel @n2 [1]
func.func @sibling_not_contiguous() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %b = memref.alloc() : memref<640xbf16, 1>
      air.channel.put @n2[%c0] (%b[0] [256] [1]) : (memref<640xbf16, 1>)
      air.channel.put @n2[%c0] (%b[384] [256] [1]) : (memref<640xbf16, 1>)
      memref.dealloc %b : memref<640xbf16, 1>
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: adjacent siblings on DIFFERENT channels are two transfers,
// not one written twice. They tile the buffer perfectly and folding on offsets
// alone would merge them -- and send both halves down the first channel, so
// the second channel's consumer waits forever.

// CHECK-LABEL: func.func @sibling_two_channels
// CHECK: air.channel.put{{.*}}@t1[%c0] (%results[0] [256] [1])
// CHECK: air.channel.put{{.*}}@t2[%c0] (%results[256] [256] [1])
air.channel @t1 [1]
air.channel @t2 [1]
func.func @sibling_two_channels() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %b = memref.alloc() : memref<512xbf16, 1>
      air.channel.put @t1[%c0] (%b[0] [256] [1]) : (memref<512xbf16, 1>)
      air.channel.put @t2[%c0] (%b[256] [256] [1]) : (memref<512xbf16, 1>)
      memref.dealloc %b : memref<512xbf16, 1>
    }
  }
  return
}
