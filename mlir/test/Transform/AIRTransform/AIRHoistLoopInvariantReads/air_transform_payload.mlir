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

