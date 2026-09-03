//===- dma_to_channel_external_pattern.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file -verify-diagnostics | FileCheck %s

// The far side's window, written in the far side's loop.
//
// One op naming both endpoints can only describe a window it can NAME. A
// memtile feed steps its source once per chunk while the core consuming it runs
// a different loop entirely, so the producer's offset is a function of the
// producer's induction variable -- and there is no such value where the core
// sits. Spelled as channel ops that is two ops in two nests and says itself;
// spelled as one op it could not be said at all, and the transfer was simply
// not portable.
//
// `external_offsets` is that missing sentence: a map over the ANCHOR's
// enclosing loops, innermost first. d0 is the loop the transfer lands in.
// Binding the window to those induction variables is also what gives the
// external half its MULTIPLICITY -- it iterates with the anchor's loop, not
// with the op that wrote it -- so one attribute answers both halves of what a
// single op otherwise cannot express.

// CHECK-LABEL: func.func @producer_indexed_window
// The external half lands in the producer's loop and steps with it.
// CHECK: scf.for %[[R:.*]] = %c0
// CHECK: air.channel.get{{.*}}@f
// CHECK: %[[OFF:.*]] = affine.apply {{.*}}(%[[R]])
// CHECK: air.channel.put{{.*}}@x{{.*}}[%[[OFF]]] [256] [1]
// The consumer keeps the window it wrote for itself.
// CHECK: air.channel.get{{.*}}@x{{.*}}[] [] []

air.channel @x [1]
air.channel @f [1]
func.func @producer_indexed_window(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      scf.for %r = %c0 to %c2 step %c1_s {
        air.channel.get @f[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x, channel_indices = array<i64: 0>, hoist_after = @f, external_offsets = affine_map<(d0) -> (d0 * 256)>, external_sizes = array<i64: 256>, external_strides = array<i64: 1>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: without an anchor there is no loop to write the map
// against, so the map names induction variables that do not exist anywhere the
// transfer could land. Reject rather than silently drop it.

air.channel @x2 [1]
func.func @no_anchor(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c1_s = arith.constant 1 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        // expected-error @+2 {{carries external_offsets but names no anchor}}
        // expected-error @+1 {{failed to legalize operation 'air.dma_memcpy_nd'}}
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x2, channel_indices = array<i64: 0>, external_offsets = affine_map<(d0) -> (d0 * 256)>, external_sizes = array<i64: 256>, external_strides = array<i64: 1>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: a partial access pattern has no meaning. Offsets alone say
// where but not how much, and the sizes cannot be inferred from the near side
// -- the two halves of a transfer are allowed to differ in extent, which is the
// whole reason a header word can be stripped in flight.

air.channel @x3 [1]
air.channel @f3 [1]
func.func @partial_pattern(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      scf.for %r = %c0 to %c2 step %c1_s {
        air.channel.get @f3[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        // expected-error @+2 {{carries external_offsets without external_sizes and external_strides}}
        // expected-error @+1 {{failed to legalize operation 'air.dma_memcpy_nd'}}
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x3, channel_indices = array<i64: 0>, hoist_after = @f3, external_offsets = affine_map<(d0) -> (d0 * 256)>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: a map written over more loops than the anchor sits in. The
// anchor decides the depth, so a map assuming more is a front-end error about
// where the transfer will land, not something to silently truncate.

air.channel @x4 [1]
air.channel @f4 [1]
func.func @map_too_deep(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      // the anchor sits in NO loop
      air.channel.get @f4[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        // This one is caught after the split, when the transfer has landed and
        // its real depth is known, so there is no legalization cascade.
        // expected-error @+1 {{external_offsets is written over 1 enclosing loop(s), but the transfer landed in 0}}
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x4, channel_indices = array<i64: 0>, hoist_after = @f4, external_offsets = affine_map<(d0) -> (d0 * 256)>, external_sizes = array<i64: 256>, external_strides = array<i64: 1>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}
