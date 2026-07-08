//===- air_annotate_refeed.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-refeed --split-input-file --verify-diagnostics | FileCheck %s

// Opt-in producer: an scf.for tagged air.refeed_loop whose body is a single
// loop-invariant channel.put is collapsed to one put, and its trip count is
// recorded as air.refeed_count on the channel declaration (the authoritative
// carrier the lock allocators read).

// CHECK: air.channel @c0 [1, 1] {air.refeed_count = 4 : i32}
// CHECK-LABEL: @scf_refeed
// CHECK-NOT: scf.for
// CHECK: air.channel.put @c0[] (%{{.*}}[] [] []) : (memref<64xi32, 2>)
air.channel @c0 [1, 1]
func.func @scf_refeed(%m: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c0[] (%m[] [] []) : (memref<64xi32, 2>)
  } {air.refeed_loop}
  return
}

// -----

// affine.for variant.

// CHECK: air.channel @c1 [1, 1] {air.refeed_count = 8 : i32}
// CHECK-LABEL: @affine_refeed
// CHECK-NOT: affine.for
// CHECK: air.channel.put @c1[]
air.channel @c1 [1, 1]
func.func @affine_refeed(%m: memref<64xi32, 2>) {
  affine.for %i = 0 to 8 {
    air.channel.put @c1[] (%m[] [] []) : (memref<64xi32, 2>)
  } {air.refeed_loop}
  return
}

// -----

// Existing air.refeed_count on the channel is raised to max(existing, N).

// CHECK: air.channel @c2 [1, 1] {air.refeed_count = 4 : i32}
air.channel @c2 [1, 1] {air.refeed_count = 3 : i32}
func.func @max_refeed(%m: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c2[] (%m[] [] []) : (memref<64xi32, 2>)
  } {air.refeed_loop}
  return
}

// -----

// Unsafe: the put offset depends on the induction variable (NOT invariant), so
// the loop is left unchanged and no refeed_count is recorded.

// CHECK: air.channel @c3 [1, 1]{{$}}
// CHECK-LABEL: @not_invariant
// CHECK: scf.for
air.channel @c3 [1, 1]
func.func @not_invariant(%m: memref<256xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // expected-warning @+1 {{channel.put is not loop-invariant}}
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c3[] (%m[%i] [%c4] [%c1]) : (memref<256xi32, 2>)
  } {air.refeed_loop}
  return
}

// -----

// Unsafe: body is more than a single channel.put, so it is left unchanged.

// CHECK: air.channel @c4 [1, 1]{{$}}
// CHECK-LABEL: @not_single_put
// CHECK: scf.for
air.channel @c4 [1, 1]
func.func @not_single_put(%m: memref<64xi32, 2>, %n: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // expected-warning @+1 {{body is not a single channel.put}}
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c4[] (%m[] [] []) : (memref<64xi32, 2>)
    air.channel.put @c4[] (%n[] [] []) : (memref<64xi32, 2>)
  } {air.refeed_loop}
  return
}
