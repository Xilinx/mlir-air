//===- specialize-nonconstant-wrap.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride=scope=func | FileCheck %s

// A hardware buffer descriptor has a constant shape per dimension, so a loop
// whose channel op carries an IV-dependent wrap SIZE (e.g. a causal-triangle
// (lx+1)*N transfer) cannot be folded into a single strided descriptor. The
// pass must decline the fold gracefully (not crash) and let the loop unroll
// into a chain of constant-shape descriptors.

#tri = affine_map<(d0) -> ((d0 + 1) * 8)>

air.channel @K [1]
air.channel @Rect [1, 1]
air.channel @Kaff [1]

// CHECK-LABEL: @triangle
// CHECK-NOT: scf.for
// CHECK-COUNT-4: air.channel.put @K
// CHECK-NOT: air.channel.put
func.func @triangle(%arg0: memref<2048x512xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  scf.for %lx = %c0 to %c4 step %c1 {
    %sz = affine.apply #tri(%lx)
    air.channel.put @K[%c0] (%arg0[%c0, %c0] [%sz, %c64, %c64] [%c512, %c512, %c1]) : (memref<2048x512xbf16>)
  }
  return
}

// A rectangular loop (IV in the data OFFSET, constant wrap) is unaffected: it
// still folds into a single strided descriptor.

// CHECK-LABEL: @rectangular
// CHECK-NOT: scf.for
// CHECK-COUNT-1: air.channel.put @Rect
// CHECK-NOT: air.channel.put
func.func @rectangular(%arg0: memref<512x64xbf16, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  scf.for %j = %c0 to %c4 step %c1 {
    %off = arith.muli %j, %c16 : index
    air.channel.put @Rect[%c0, %c0] (%arg0[%off, %c0] [%c8, %c8] [%c64, %c1]) : (memref<512x64xbf16, 1>)
  }
  return
}

// The affine.for twin of @triangle must behave identically (its own specialize
// pattern also declines the fold on a non-constant wrap and unrolls).

// CHECK-LABEL: @triangle_affine
// CHECK-NOT: affine.for
// CHECK-COUNT-4: air.channel.put @Kaff
// CHECK-NOT: air.channel.put
func.func @triangle_affine(%arg0: memref<2048x512xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  affine.for %lx = 0 to 4 {
    %sz = affine.apply #tri(%lx)
    air.channel.put @Kaff[%c0] (%arg0[%c0, %c0] [%sz, %c64, %c64] [%c512, %c512, %c1]) : (memref<2048x512xbf16>)
  }
  return
}
