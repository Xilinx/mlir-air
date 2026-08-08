//===- unroll-bd-chain-lock-limit.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// AIRUnrollScfForIntoBDChain declines to unroll a loop whose body is a single
// channel op with no enclosing IV in its OFFSETS: every unrolled copy would be
// identical, and downstream air-to-aie would set the lock init count to the
// trip count, breaking per-iteration pairing. Leaving the loop intact keeps
// init=1 with an implicit per-iteration BD repeat.
//
// The guard fires when the trip count REACHES the per-channel lock capacity
// (16), not merely when it exceeds it -- a loop of exactly 16 is already at
// capacity. Both functions below have an IV-dependent transfer SIZE, which is
// what makes wrap-and-stride decline to fold them and hands them to the unroll
// pattern in the first place (a loop-invariant put folds earlier and never gets
// here). @below_limit is under the cap and unrolls; @at_limit sits exactly on
// it and must NOT.

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride="scope=func" | FileCheck %s

// Trip count 8: below the cap, so the loop is unrolled into one op per
// iteration and no scf.for survives.
// CHECK-LABEL: @below_limit
// CHECK-NOT: scf.for
// CHECK-COUNT-8: air.channel.put
// CHECK-NOT: air.channel.put

// Trip count 16: exactly at the cap, so the loop is kept and emits a single
// channel op. With a strictly-greater-than guard this unrolls into 16.
// CHECK-LABEL: @at_limit
// CHECK: scf.for
// CHECK-COUNT-1: air.channel.put
// CHECK-NOT: air.channel.put

#map = affine_map<(d0)[] -> (d0 * 8 + 8)>
module {
  air.channel @ch8 [1, 1]
  air.channel @ch16 [1, 1]

  func.func @below_limit(%buf: memref<1024xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %sz = affine.apply #map(%i)[]
      air.channel.put @ch8[] (%buf[%c0] [%sz] [%c1]) : (memref<1024xi32, 1>)
    }
    return
  }

  func.func @at_limit(%buf: memref<1024xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %sz = affine.apply #map(%i)[]
      air.channel.put @ch16[] (%buf[%c0] [%sz] [%c1]) : (memref<1024xi32, 1>)
    }
    return
  }
}
