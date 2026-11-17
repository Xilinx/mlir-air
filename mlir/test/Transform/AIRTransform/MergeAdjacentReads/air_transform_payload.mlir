//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s

// Two rows of 16 bf16, one after the other: one read of 32, its halves.
// CHECK-LABEL: @halves
// CHECK: %[[W:.*]] = vector.transfer_read %arg0[%c0, %c0], %{{.*}} {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<2x16xbf16>
// CHECK: %[[F:.*]] = vector.shape_cast %[[W]] : vector<2x16xbf16> to vector<32xbf16>
// CHECK: vector.extract_strided_slice %[[F]] offsets = [0], sizes = [16], strides = [1]
// CHECK: vector.extract_strided_slice %[[F]] offsets = [16], sizes = [16], strides = [1]
// CHECK-NOT: vector.transfer_read
func.func @halves(%m: memref<2x16xbf16, 2>) -> (vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b : vector<1x16xbf16>, vector<1x16xbf16>
}

// Two rows of 32 bf16 would make 1024 bits, over max_bits: left alone.
// CHECK-LABEL: @too_wide
// CHECK-COUNT-2: vector.transfer_read {{.*}} vector<1x32xbf16>
func.func @too_wide(%m: memref<2x32xbf16, 2>) -> (vector<1x32xbf16>, vector<1x32xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x32xbf16, 2>, vector<1x32xbf16>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x32xbf16, 2>, vector<1x32xbf16>
  return %a, %b : vector<1x32xbf16>, vector<1x32xbf16>
}

// Rows 0 and 2 are not adjacent; half rows do not cover their dim.
// CHECK-LABEL: @not_adjacent
// CHECK-COUNT-4: vector.transfer_read
func.func @not_adjacent(%m: memref<4x16xbf16, 2>, %n: memref<2x32xbf16, 2>) -> (vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %p = arith.constant 0.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<4x16xbf16, 2>, vector<1x16xbf16>
  %b = vector.transfer_read %m[%c2, %c0], %p {in_bounds = [true, true]} : memref<4x16xbf16, 2>, vector<1x16xbf16>
  %c = vector.transfer_read %n[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x32xbf16, 2>, vector<1x16xbf16>
  %d = vector.transfer_read %n[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x32xbf16, 2>, vector<1x16xbf16>
  return %a, %b, %c, %d : vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>
}

// A write to the memory between the reads keeps them apart; a write to
// another allocation does not.
// CHECK-LABEL: @writes_between
// CHECK: vector.transfer_read %[[M:.*]][%c0, %c0], {{.*}} vector<1x16xbf16>
// CHECK: vector.transfer_write {{.*}}, %[[M]]
// CHECK: vector.transfer_read %[[M]][%c1, %c0], {{.*}} vector<1x16xbf16>
// CHECK: vector.transfer_read %[[O:.*]][%c0, %c0], {{.*}} vector<2x16xbf16>
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read %[[O]]
func.func @writes_between(%v: vector<1x16xbf16>) -> (vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %m = memref.alloc() : memref<2x16xbf16, 2>
  %o = memref.alloc() : memref<2x16xbf16, 2>
  %x = memref.alloc() : memref<2x16xbf16, 2>
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  vector.transfer_write %v, %m[%c1, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<2x16xbf16, 2>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %c = vector.transfer_read %o[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  vector.transfer_write %v, %x[%c1, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<2x16xbf16, 2>
  %d = vector.transfer_read %o[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b, %c, %d : vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>, vector<1x16xbf16>
}

// In a herd, its arguments are the allocations passed to it: a write
// through the argument of another allocation does not keep the reads apart,
// one through the argument of the same allocation does.
// CHECK-LABEL: @herd_args
// CHECK: air.herd
// CHECK: vector.transfer_read %[[A:.*]][%c0, %c0], {{.*}} vector<2x16xbf16>
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read %[[A]]
// CHECK: vector.transfer_read %[[C:.*]][%c0, %c0], {{.*}} vector<1x16xbf16>
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read %[[C]][%{{.*}}, %c0], {{.*}} vector<1x16xbf16>
func.func @herd_args(%v: vector<1x16xbf16>) {
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<2x16xbf16, 2>
  %b = memref.alloc() : memref<2x16xbf16, 2>
  air.herd @h tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%ha=%a, %hb=%b, %hv=%v, %hc=%a) : memref<2x16xbf16, 2>, memref<2x16xbf16, 2>, vector<1x16xbf16>, memref<2x16xbf16, 2> {
    %c0 = arith.constant 0 : index
    %c1h = arith.constant 1 : index
    %p = arith.constant 0.0 : bf16
    %r0 = vector.transfer_read %ha[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
    vector.transfer_write %hv, %hb[%c1h, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<2x16xbf16, 2>
    %r1 = vector.transfer_read %ha[%c1h, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
    %s0 = vector.transfer_read %hc[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
    vector.transfer_write %r0, %ha[%c1h, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<2x16xbf16, 2>
    %s1 = vector.transfer_read %hc[%c1h, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
    %t0 = arith.addf %r1, %s0 : vector<1x16xbf16>
    %t1 = arith.addf %t0, %s1 : vector<1x16xbf16>
    vector.transfer_write %t1, %hb[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xbf16>, memref<2x16xbf16, 2>
  }
  return
}
