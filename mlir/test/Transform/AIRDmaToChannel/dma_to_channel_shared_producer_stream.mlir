//===- dma_to_channel_shared_producer_stream.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// Several consumer sites SHARING one producer stream must derive ONE producer.
//
// A memtile fill feeding a herd that expands its inner loop more than once -- a
// GEMV emitted per pair row, say -- reaches the hoist once per site. Each site
// names the same channel, the same memtile buffer and the same window, and each
// is anchored to the same feed, so each used to clone a descriptor of its own.
// That second descriptor is not a second transfer: it re-sends bytes the first
// already sent, on the same sub-channel, acquiring the fill lock a second time
// for a fill that happens once. The memtile MM2S ring then carries two acquires
// per release against an S2MM that still has one, and the design reads stale
// data -- it compiles, stays under the BD budget, and is simply wrong.
//
// The front end could not have said this differently. Spelled as channel ops it
// writes ONE put in the producer's own loop and lets several gets share it,
// which is legal, and is the shape restored here: one put, N gets.
//
// The comparison has to be by VALUE, not SSA identity. Every hoisted transfer
// materializes its own copy of the pure defs it needs -- that is what lets a
// clone stand alone where it lands -- so two spellings of one constant offset
// are different Values by construction.

// CHECK-LABEL: func.func @shared_stream
// One derived put, at the anchor inside the producer's loop...
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@f
// CHECK: air.channel.put{{.*}}@w
// CHECK-NOT: air.channel.put{{.*}}@w
// ...and both consumer sites keep their own get, sharing it.
// CHECK: air.channel.get{{.*}}@w
// CHECK: air.channel.get{{.*}}@w

air.channel @w [1]
air.channel @f [1]
func.func @shared_stream(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      // The producer: one fill per step; the fan is derived from the herd.
      %l2 = memref.alloc() : memref<64xbf16, 1>
      scf.for %r = %c0 to %c8 step %c1_s {
        air.channel.get @f[%c0] (%l2[] [] []) : (memref<64xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<64xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c64_h = arith.constant 64 : index
        // Two consumer sites: same channel, same buffer, same window.
        %a = memref.alloc() : memref<64xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[%c0_h] [%c64_h] [%c1_h]) {id = 1 : i32, channel = @w, channel_indices = array<i64: 0>, hoist_after = @f} : (memref<64xbf16, 2>, memref<64xbf16, 1>)
        memref.dealloc %a : memref<64xbf16, 2>
        %a2 = memref.alloc() : memref<64xbf16, 2>
        air.dma_memcpy_nd (%a2[] [] [], %b[%c0_h] [%c64_h] [%c1_h]) {id = 2 : i32, channel = @w, channel_indices = array<i64: 0>, hoist_after = @f} : (memref<64xbf16, 2>, memref<64xbf16, 1>)
        memref.dealloc %a2 : memref<64xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: two sites reading DIFFERENT windows of the same buffer are
// two transfers, and both descriptors must survive. This is the ordinary fan --
// a memtile handing each consumer its own slice -- and collapsing it would
// starve every consumer but the first.

// CHECK-LABEL: func.func @distinct_windows
// CHECK: air.channel.put{{.*}}@w2
// CHECK: air.channel.put{{.*}}@w2

air.channel @w2 [1]
air.channel @f2 [1]
func.func @distinct_windows(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %l2 = memref.alloc() : memref<64xbf16, 1>
      scf.for %r = %c0 to %c8 step %c1_s {
        air.channel.get @f2[%c0] (%l2[] [] []) : (memref<64xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<64xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c32_h = arith.constant 32 : index
        %a = memref.alloc() : memref<32xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[%c0_h] [%c32_h] [%c1_h]) {id = 1 : i32, channel = @w2, channel_indices = array<i64: 0>, hoist_after = @f2} : (memref<32xbf16, 2>, memref<64xbf16, 1>)
        memref.dealloc %a : memref<32xbf16, 2>
        %a2 = memref.alloc() : memref<32xbf16, 2>
        air.dma_memcpy_nd (%a2[] [] [], %b[%c32_h] [%c32_h] [%c1_h]) {id = 2 : i32, channel = @w2, channel_indices = array<i64: 0>, hoist_after = @f2} : (memref<32xbf16, 2>, memref<64xbf16, 1>)
        memref.dealloc %a2 : memref<32xbf16, 2>
      }
    }
  }
  return
}
