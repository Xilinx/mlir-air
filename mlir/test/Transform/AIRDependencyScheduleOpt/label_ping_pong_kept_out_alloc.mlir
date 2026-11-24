//===- label_ping_pong_kept_out_alloc.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong --split-input-file | FileCheck %s

// air-fuse-alloc-dealloc keeps out of a loop the buffers it cannot show carry
// nothing across iterations. The labeler judges them as it judges the allocs
// in the body, and moves them in only for a loop it labels.

// The callee reaches the accumulator first and may read what the previous
// iteration left in it: the loop is not labelled and the alloc stays out.

// CHECK-LABEL: @kept_out_live
// CHECK: memref.alloc() : memref<16xf32, 2>
// CHECK: scf.for
// CHECK-NOT: memref.alloc() : memref<16xf32, 2>
// CHECK-NOT: unroll
air.channel @c0 [1, 1]
func.func private @accumulate(memref<64xbf16, 2>, memref<16xf32, 2>)
func.func @kept_out_live() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t1, %acc = air.execute -> (memref<16xf32, 2>) {
    %alloc = memref.alloc() : memref<16xf32, 2>
    air.execute_terminator %alloc : memref<16xf32, 2>
  }
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t1) -> (!air.async.token) {
    %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
      %alloc = memref.alloc() : memref<64xbf16, 2>
      air.execute_terminator %alloc : memref<64xbf16, 2>
    }
    %g = air.channel.get async [%tok0] @c0[] (%buf[] [] []) : (memref<64xbf16, 2>)
    %tok2 = air.execute [%g] {
      func.call @accumulate(%buf, %acc) : (memref<64xbf16, 2>, memref<16xf32, 2>) -> ()
    }
    %tok3 = air.execute [%tok2] {
      memref.dealloc %buf : memref<64xbf16, 2>
    }
    scf.yield %tok3 : !air.async.token
  }
  %tok4 = air.execute [%r] {
    memref.dealloc %acc : memref<16xf32, 2>
  }
  return
}

// -----

// A function the module defines overwrites the buffer before anything reads
// it in an iteration: the loop is labelled, and the buffer is moved in and
// duplicated with the others.

// CHECK-LABEL: @kept_out_dead
// CHECK: scf.for
// CHECK: memref.alloc() {hoist_alloc = true} : memref<64xbf16, 2>
// CHECK: } {unroll = 2 : i32}
air.channel @c1 [1, 1]
func.func @fill(%m: memref<64xbf16, 2>) {
  %c0 = arith.constant 0 : index
  %v = arith.constant dense<0.000000e+00> : vector<64xbf16>
  vector.transfer_write %v, %m[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16, 2>
  return
}
func.func @kept_out_dead() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t1, %buf = air.execute -> (memref<64xbf16, 2>) {
    %alloc = memref.alloc() : memref<64xbf16, 2>
    air.execute_terminator %alloc : memref<64xbf16, 2>
  }
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t1) -> (!air.async.token) {
    %tok2 = air.execute [%tok] {
      func.call @fill(%buf) : (memref<64xbf16, 2>) -> ()
    }
    %p = air.channel.put async [%tok2] @c1[] (%buf[] [] []) : (memref<64xbf16, 2>)
    scf.yield %p : !air.async.token
  }
  %tok4 = air.execute [%r] {
    memref.dealloc %buf : memref<64xbf16, 2>
  }
  return
}

// -----

// The first access stores one element; the put then reads all four, three of
// them left by earlier iterations. A write, but not of the whole buffer: the
// loop is not labelled and the alloc stays out.

// CHECK-LABEL: @kept_out_partial_write
// CHECK: memref.alloc() : memref<4xi32, 2>
// CHECK: scf.for
// CHECK-NOT: memref.alloc() : memref<4xi32, 2>
// CHECK-NOT: unroll
air.channel @c2 [1, 1]
func.func @kept_out_partial_write() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %t1, %buf = air.execute -> (memref<4xi32, 2>) {
    %alloc = memref.alloc() : memref<4xi32, 2>
    air.execute_terminator %alloc : memref<4xi32, 2>
  }
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%tok = %t1) -> (!air.async.token) {
    %tok2 = air.execute [%tok] {
      %v = arith.index_cast %i : index to i32
      memref.store %v, %buf[%i] : memref<4xi32, 2>
    }
    %p = air.channel.put async [%tok2] @c2[] (%buf[] [] []) : (memref<4xi32, 2>)
    scf.yield %p : !air.async.token
  }
  %tok4 = air.execute [%r] {
    memref.dealloc %buf : memref<4xi32, 2>
  }
  return
}

// -----

// The defined function overwrites the buffer, but only when %q holds: the loop
// is not labelled and the alloc stays out.

// CHECK-LABEL: @kept_out_conditional_overwrite
// CHECK: memref.alloc() : memref<64xbf16, 2>
// CHECK: scf.for
// CHECK-NOT: memref.alloc() : memref<64xbf16, 2>
// CHECK-NOT: unroll
air.channel @c3 [1, 1]
func.func @fill2(%m: memref<64xbf16, 2>) {
  %c0 = arith.constant 0 : index
  %v = arith.constant dense<0.000000e+00> : vector<64xbf16>
  vector.transfer_write %v, %m[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16, 2>
  return
}
func.func @kept_out_conditional_overwrite(%q: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t1, %buf = air.execute -> (memref<64xbf16, 2>) {
    %alloc = memref.alloc() : memref<64xbf16, 2>
    air.execute_terminator %alloc : memref<64xbf16, 2>
  }
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t1) -> (!air.async.token) {
    %tok2 = air.execute [%tok] {
      scf.if %q {
        func.call @fill2(%buf) : (memref<64xbf16, 2>) -> ()
      }
    }
    %p = air.channel.put async [%tok2] @c3[] (%buf[] [] []) : (memref<64xbf16, 2>)
    scf.yield %p : !air.async.token
  }
  %tok4 = air.execute [%r] {
    memref.dealloc %buf : memref<64xbf16, 2>
  }
  return
}
