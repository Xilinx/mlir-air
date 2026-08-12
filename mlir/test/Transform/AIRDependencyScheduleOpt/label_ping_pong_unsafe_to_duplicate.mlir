//===- label_ping_pong_unsafe_to_duplicate.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong --split-input-file | FileCheck %s

// The transform duplicates exactly the allocs it labels; every other memref the
// body touches stays single. These two shapes break under that, so the labeler
// declines them instead of making the design opt out by hand.

// Positive control: the alloc is filled by a channel.get before the callee sees
// it, so nothing carries into the next iteration. Labeled.

// CHECK-LABEL: @dead_on_entry
// CHECK: memref.alloc() {hoist_alloc = true} : memref<64xbf16, 2>
// CHECK: } {unroll = 2 : i32}
air.channel @c0 [1, 1]
func.func private @use(memref<64xbf16, 2>)
func.func @dead_on_entry() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t = air.wait_all async
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
      %alloc = memref.alloc() : memref<64xbf16, 2>
      air.execute_terminator %alloc : memref<64xbf16, 2>
    }
    %g = air.channel.get async [%tok0] @c0[] (%buf[] [] []) : (memref<64xbf16, 2>)
    %tok1 = air.execute [%g] {
      func.call @use(%buf) : (memref<64xbf16, 2>) -> ()
    }
    %tok2 = air.execute [%tok1] {
      memref.dealloc %buf : memref<64xbf16, 2>
    }
    scf.yield %tok2 : !air.async.token
  }
  return
}

// -----

// An opaque callee reaches the alloc first, so it may read what the previous
// iteration left there. Duplicating it would give each parity its own half of
// the accumulation. (Accumulators arrive here sunk into the body by
// air-fuse-alloc-dealloc, which is why they look per-iteration.)

// CHECK-LABEL: @live_across_iterations
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
air.channel @c1 [1, 1]
func.func private @accumulate(memref<64xbf16, 2>, memref<16xf32, 2>)
func.func @live_across_iterations() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t = air.wait_all async
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
      %alloc = memref.alloc() : memref<64xbf16, 2>
      air.execute_terminator %alloc : memref<64xbf16, 2>
    }
    %tok1, %acc = air.execute -> (memref<16xf32, 2>) {
      %alloc = memref.alloc() : memref<16xf32, 2>
      air.execute_terminator %alloc : memref<16xf32, 2>
    }
    %g = air.channel.get async [%tok0] @c1[] (%buf[] [] []) : (memref<64xbf16, 2>)
    %tok2 = air.execute [%g, %tok1] {
      func.call @accumulate(%buf, %acc) : (memref<64xbf16, 2>, memref<16xf32, 2>) -> ()
    }
    %tok3 = air.execute [%tok2] {
      memref.dealloc %buf : memref<64xbf16, 2>
    }
    %tok4 = air.execute [%tok2] {
      memref.dealloc %acc : memref<16xf32, 2>
    }
    scf.yield %tok3 : !air.async.token
  }
  return
}

// -----

// The score buffer enters as a herd argument, so it is shared with the herd's
// other tiles and the transform leaves it single. The handshake around it is
// taken once per body; running the body twice per handshake races it.

// CHECK-LABEL: @writes_shared_herd_arg
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
air.channel @c2 [1, 1]
func.func private @score(memref<64xbf16, 2>, memref<32xbf16, 2>)
func.func @writes_shared_herd_arg(%shared: memref<32xbf16, 2>) {
  %c1 = arith.constant 1 : index
  air.herd @h tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%s=%shared) : memref<32xbf16, 2> {
    %h0 = arith.constant 0 : index
    %h1 = arith.constant 1 : index
    %h8 = arith.constant 8 : index
    %t = air.wait_all async
    %r = scf.for %i = %h0 to %h8 step %h1 iter_args(%tok = %t) -> (!air.async.token) {
      %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
        %alloc = memref.alloc() : memref<64xbf16, 2>
        air.execute_terminator %alloc : memref<64xbf16, 2>
      }
      %g = air.channel.get async [%tok0] @c2[] (%buf[] [] []) : (memref<64xbf16, 2>)
      %tok1 = air.execute [%g] {
        func.call @score(%buf, %s) : (memref<64xbf16, 2>, memref<32xbf16, 2>) -> ()
      }
      %tok2 = air.execute [%tok1] {
        memref.dealloc %buf : memref<64xbf16, 2>
      }
      scf.yield %tok2 : !air.async.token
    }
  }
  return
}

// -----

// Same herd, same loop, but the callee only sees the per-iteration buffer. The
// herd wrapper by itself is not what disqualifies the case above.

// CHECK-LABEL: @herd_without_shared_arg
// CHECK: memref.alloc() {hoist_alloc = true} : memref<64xbf16, 2>
// CHECK: } {unroll = 2 : i32}
air.channel @c3 [1, 1]
func.func private @local(memref<64xbf16, 2>)
func.func @herd_without_shared_arg(%shared: memref<32xbf16, 2>) {
  %c1 = arith.constant 1 : index
  air.herd @h tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%s=%shared) : memref<32xbf16, 2> {
    %h0 = arith.constant 0 : index
    %h1 = arith.constant 1 : index
    %h8 = arith.constant 8 : index
    %t = air.wait_all async
    %r = scf.for %i = %h0 to %h8 step %h1 iter_args(%tok = %t) -> (!air.async.token) {
      %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
        %alloc = memref.alloc() : memref<64xbf16, 2>
        air.execute_terminator %alloc : memref<64xbf16, 2>
      }
      %g = air.channel.get async [%tok0] @c3[] (%buf[] [] []) : (memref<64xbf16, 2>)
      %tok1 = air.execute [%g] {
        func.call @local(%buf) : (memref<64xbf16, 2>) -> ()
      }
      %tok2 = air.execute [%tok1] {
        memref.dealloc %buf : memref<64xbf16, 2>
      }
      scf.yield %tok2 : !air.async.token
    }
  }
  return
}
