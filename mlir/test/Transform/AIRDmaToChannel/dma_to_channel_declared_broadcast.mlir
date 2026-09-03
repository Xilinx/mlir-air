//===- dma_to_channel_declared_broadcast.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-broadcast-detection -air-specialize-dma-broadcast -air-dma-to-channel -split-input-file | FileCheck %s

// A DMA naming a channel whose DECLARATION carries a broadcast_shape has
// already stated its fan-out. Deriving a second one from the enclosing herd is
// the same fact written twice, and the two do not have to agree: broadcast
// detection reads the herd the DMA happens to sit in, so a [4, 4] declaration
// read from inside a [2, 4] block herd comes back [2, 4].
//
// The cost is not only the disagreement. Specialization turns the derived
// pattern into an affine.if around the consumer, and that region is what stops
// canonicalize from folding away the ping-pong duplicates the transform
// speculatively allocates -- so a core that shared one 2-deep ring across its
// two GEMV sites gets a separate ring per site, blows its core-memory BD
// budget, and hangs on device.
//
// Two independent rules are asserted here:
//   1. an invariant transfer on a DECLARED broadcast is not re-specialized;
//   2. the index on a broadcast identifies the RECEIVER, so the producing half
//      carries none -- a herd induction variable does not exist where the
//      producer lands, and leaving it there is a dominance error.

// CHECK-LABEL: func.func @declared_broadcast
// The producer is indexed by nothing at all (it is hoisted to the segment)...
// CHECK: air.channel.put{{.*}}@bc[] (
// ...and no guard is synthesised around the consumer: the declaration already
// says "every core". This CHECK-NOT has to sit BETWEEN the two, because the
// affine.if it forbids WRAPS the get -- putting it above the put would end its
// search range before the guard is ever reached, and the control would pass
// with the rule disabled.
// CHECK-NOT: affine.if
// CHECK: air.channel.get{{.*}}@bc[%{{.*}}, %{{.*}}]
air.channel @bc [1, 1] {broadcast_shape = [2, 2]}
func.func @declared_broadcast(%arg0: memref<256xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<256xbf16> {
    air.segment @seg args(%sa=%la) : memref<256xbf16> {
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%b=%l2) : memref<256xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @bc} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
      memref.dealloc %l2 : memref<256xbf16, 1>
    }
  }
  return
}

// -----

// NEGATIVE CONTROL, and the one that matters: a transfer whose WINDOW VARIES
// with a herd index is genuinely different per core, and the affine.if is what
// turns that variance into one constant descriptor apiece. Skipping
// specialization here would leave the herd induction variable in the
// producer's offsets after the hoist -- a dominance error, not a redundancy.
// So a declaration alone must NOT be enough to decline.
//
// This is the case the first version of the rule broke.

// CHECK-LABEL: func.func @declared_broadcast_but_variant
// CHECK: affine.if
air.channel @bcv [1, 1] {broadcast_shape = [1, 2]}
#map = affine_map<()[s0] -> (s0 * 32)>
func.func @declared_broadcast_but_variant(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2) args(%a=%sa) : memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %cst1 = arith.constant 1 : index
        %off = affine.apply #map()[%tx]
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%off, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @bcv} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: an UNDECLARED broadcast still gets one derived. Nothing
// else states the fan-out, so detection is its only source and declining would
// leave the herd's cores with no guard and no shared flow.

// CHECK-LABEL: func.func @undeclared_broadcast
// CHECK: affine.if
func.func @undeclared_broadcast(%arg0: memref<256xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<256xbf16> {
    air.segment @seg args(%sa=%la) : memref<256xbf16> {
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%b=%l2) : memref<256xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
      memref.dealloc %l2 : memref<256xbf16, 1>
    }
  }
  return
}
