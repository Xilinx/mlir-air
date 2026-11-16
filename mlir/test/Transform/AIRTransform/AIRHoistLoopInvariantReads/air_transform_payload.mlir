//===- air_transform_payload.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// A read of an allocation no op in the loop writes, at indices from outside
// the loop, leaves it; one of a buffer the loop writes, or at the loop's
// index, stays.
// CHECK-LABEL: @hoist
// CHECK: %[[P:.*]] = vector.transfer_read %[[A:.*]][%c0, %c0]
// CHECK: scf.for
// CHECK-NOT: vector.transfer_read %[[A]][%c0, %c0]
// CHECK: vector.transfer_read %{{.*}}[%arg
// CHECK: vector.transfer_read %[[B:.*]][%c0, %c0]
// CHECK: vector.transfer_write {{.*}}, %[[B]]
func.func @hoist(%out: memref<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %a = memref.alloc() : memref<8x8xf32>
  %b = memref.alloc() : memref<8x8xf32>
  %w = memref.alloc() : memref<4x8xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    %p = vector.transfer_read %a[%c0, %c0], %f0 {in_bounds = [true]} : memref<8x8xf32>, vector<8xf32>
    %q = vector.transfer_read %w[%i, %c0], %f0 {in_bounds = [true]} : memref<4x8xf32>, vector<8xf32>
    %r = vector.transfer_read %b[%c0, %c0], %f0 {in_bounds = [true]} : memref<8x8xf32>, vector<8xf32>
    %s = arith.addf %p, %q : vector<8xf32>
    %t = arith.addf %s, %r : vector<8xf32>
    vector.transfer_write %t, %b[%c0, %c0] {in_bounds = [true]} : vector<8xf32>, memref<8x8xf32>
    vector.transfer_write %t, %out[%i, %c0] {in_bounds = [true]} : vector<8xf32>, memref<4x8xf32>
  }
  return
}

// A write through a loop-carried memref that starts as the read's buffer may
// write it: the read stays.
// CHECK-LABEL: @iter_arg_write
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @iter_arg_write(%v: vector<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %a = memref.alloc() : memref<8xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%buf = %a) -> (memref<8xf32>) {
    %p = vector.transfer_read %a[%c0], %f0 {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
    %s = arith.addf %p, %v : vector<8xf32>
    vector.transfer_write %s, %buf[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
    scf.yield %buf : memref<8xf32>
  }
  return
}

// A loop that may not run: the read stays in it.
// CHECK-LABEL: @maybe_zero_trips
// CHECK: scf.for
// CHECK: vector.transfer_read
func.func @maybe_zero_trips(%n: index, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %a = memref.alloc(%n) : memref<?xf32>
  scf.for %i = %c0 to %n step %c1 {
    %p = vector.transfer_read %a[%c0], %f0 {in_bounds = [true]} : memref<?xf32>, vector<8xf32>
    vector.transfer_write %p, %out[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  }
  return
}


// In a herd, its arguments are the allocations passed to it: the read of
// %ho, an allocation the loop does not write, is hoisted; the read of %ha
// stays, as %hb, written in the loop, is the same allocation.
// CHECK-LABEL: @herd_args
// CHECK: air.herd
// CHECK: %[[S:.*]] = vector.transfer_read %arg{{[0-9]+}}[%c0]
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: arith.addf {{.*}}, %[[S]]
func.func @herd_args(%v: vector<16xbf16>) {
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<16xbf16, 2>
  %o = memref.alloc() : memref<16xbf16, 2>
  air.herd @h tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%ha=%a, %hb=%a, %ho=%o, %hv=%v) : memref<16xbf16, 2>, memref<16xbf16, 2>, memref<16xbf16, 2>, vector<16xbf16> {
    %c0 = arith.constant 0 : index
    %c1h = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %p = arith.constant 0.0 : bf16
    scf.for %i = %c0 to %c4 step %c1h {
      %r = vector.transfer_read %ha[%c0], %p {in_bounds = [true]} : memref<16xbf16, 2>, vector<16xbf16>
      %s = vector.transfer_read %ho[%c0], %p {in_bounds = [true]} : memref<16xbf16, 2>, vector<16xbf16>
      %t = arith.addf %r, %s : vector<16xbf16>
      vector.transfer_write %t, %hb[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2>
    }
  }
  return
}
