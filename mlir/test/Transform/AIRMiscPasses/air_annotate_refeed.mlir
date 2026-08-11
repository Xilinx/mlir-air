//===- air_annotate_refeed.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-refeed --split-input-file --verify-diagnostics | FileCheck %s

// An N-trip loop whose body is nothing but a loop-invariant channel.put is a
// re-broadcast, not N productions. It collapses to one put and N lands on the
// carrier air-to-aie reads: the put itself for an L1 source, which is what the
// core-side lock allocator picks up.

// CHECK-LABEL: @scf_refeed
// CHECK-NOT: scf.for
// CHECK: air.channel.put @c0[] (%{{.*}}[] [] []) {air.refeed_count = 4 : i32}
air.channel @c0 [1, 1]
func.func @scf_refeed(%m: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c0[] (%m[] [] []) : (memref<64xi32, 2>)
  }
  return
}

// -----

// affine.for variant.

// CHECK-LABEL: @affine_refeed
// CHECK-NOT: affine.for
// CHECK: air.channel.put @c1[]{{.*}}{air.refeed_count = 8 : i32}
air.channel @c1 [1, 1]
func.func @affine_refeed(%m: memref<64xi32, 2>) {
  affine.for %i = 0 to 8 {
    air.channel.put @c1[] (%m[] [] []) : (memref<64xi32, 2>)
  }
  return
}

// -----

// An L2 source puts the count on the backing memref.alloc instead: that is the
// carrier AllocL2BuffersPattern propagates onto the buffer op, which is the
// fill/drain rendezvous the memtile lock allocator reads.

// CHECK-LABEL: @l2_carrier
// CHECK: memref.alloc() {air.refeed_count = 4 : i32}
// CHECK-NOT: scf.for
// CHECK: air.channel.put
// CHECK-NOT: air.refeed_count
air.channel @c2 [1, 1]
func.func @l2_carrier() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %buf = memref.alloc() : memref<64xi32, 1>
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c2[] (%buf[] [] []) : (memref<64xi32, 1>)
  }
  memref.dealloc %buf : memref<64xi32, 1>
  return
}

// -----

// Two re-feed loops draining the same resident L2 buffer ADD. air-to-aie makes
// the fill release the count once per fill, so 3 drains plus 4 drains need 7
// enabled -- not 12, which a multiplying accumulator would give.

// CHECK-LABEL: @counts_add
// CHECK: memref.alloc() {air.refeed_count = 7 : i32}
// CHECK-NOT: scf.for
air.channel @c3 [1, 1]
air.channel @c3b [1, 1]
func.func @counts_add() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %buf = memref.alloc() : memref<64xi32, 1>
  scf.for %i = %c0 to %c3 step %c1 {
    air.channel.put @c3[] (%buf[] [] []) : (memref<64xi32, 1>)
  }
  scf.for %j = %c0 to %c4 step %c1 {
    air.channel.put @c3b[] (%buf[] [] []) : (memref<64xi32, 1>)
  }
  memref.dealloc %buf : memref<64xi32, 1>
  return
}

// -----

// Nested re-feed loops COMPOSE. The walk is post-order, so the inner loop folds
// first and hoists the put into the outer body, where it matches again. The
// same put re-folded means M x N sends, not M + N -- priming the fill lock for
// 7 when the buffer is drained 12 times under-primes it and deadlocks.

// CHECK-LABEL: @nested_composes
// CHECK: memref.alloc() {air.refeed_count = 12 : i32}
// CHECK-NOT: scf.for
air.channel @c3c [1, 1]
func.func @nested_composes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %buf = memref.alloc() : memref<64xi32, 1>
  scf.for %i = %c0 to %c3 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      air.channel.put @c3c[] (%buf[] [] []) : (memref<64xi32, 1>)
    }
  }
  memref.dealloc %buf : memref<64xi32, 1>
  return
}

// -----

// A nest and a sibling on one buffer: 2 x 3 sends plus 4 sends is 10 drains.

// CHECK-LABEL: @nest_plus_sibling
// CHECK: memref.alloc() {air.refeed_count = 10 : i32}
// CHECK-NOT: scf.for
air.channel @c3d [1, 1]
air.channel @c3e [1, 1]
func.func @nest_plus_sibling() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %buf = memref.alloc() : memref<64xi32, 1>
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c3 step %c1 {
      air.channel.put @c3d[] (%buf[] [] []) : (memref<64xi32, 1>)
    }
  }
  scf.for %k = %c0 to %c4 step %c1 {
    air.channel.put @c3e[] (%buf[] [] []) : (memref<64xi32, 1>)
  }
  memref.dealloc %buf : memref<64xi32, 1>
  return
}

// -----

// Async form: the loop carries the put's token. The hoisted put inherits the
// loop's incoming dependency and takes over its uses, so no edge is dropped.

// CHECK-LABEL: @async
// CHECK: %[[T:.*]], %[[B:.*]] = air.execute
// CHECK-NOT: scf.for
// CHECK: %[[P:.*]] = air.channel.put async [%[[T]]]{{.*}}{air.refeed_count = 3 : i32}
// CHECK: air.execute [%[[P]]]
air.channel @c4 [1, 1]
func.func @async() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %t, %buf = air.execute -> (memref<64xi32, 2>) {
    %a = memref.alloc() : memref<64xi32, 2>
    air.execute_terminator %a : memref<64xi32, 2>
  }
  %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %p = air.channel.put async [%tok] @c4[] (%buf[] [] []) : (memref<64xi32, 2>)
    scf.yield %p : !air.async.token
  }
  %d = air.execute [%r] {
    memref.dealloc %buf : memref<64xi32, 2>
  }
  return
}

// -----

// Pure ops in the body are tolerated and hoisted with the put: a front end
// materializes the put's index constants at its own insertion point, i.e.
// inside the loop, and no canonicalizer need have run before this pass.

// CHECK-LABEL: @pure_ops_in_body
// CHECK-NOT: scf.for
// CHECK: air.channel.put @c5[]{{.*}}{air.refeed_count = 5 : i32}
air.channel @c5 [1, 1]
func.func @pure_ops_in_body(%m: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c0 to %c5 step %c1 {
    %z = arith.constant 0 : index
    %n = arith.constant 64 : index
    %s = arith.constant 1 : index
    air.channel.put @c5[] (%m[%z] [%n] [%s]) : (memref<64xi32, 2>)
  }
  return
}

// -----

// Not a re-broadcast: the put offset depends on the induction variable, so each
// trip sends different bytes. Left unchanged.

// CHECK-LABEL: @not_invariant
// CHECK: scf.for
// CHECK-NOT: air.refeed_count
air.channel @c6 [1, 1]
func.func @not_invariant(%m: memref<256xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c6[] (%m[%i] [%c4] [%c1]) : (memref<256xi32, 2>)
  }
  return
}

// -----

// Not a re-broadcast: two puts in the body. Left unchanged.

// CHECK-LABEL: @two_puts
// CHECK: scf.for
// CHECK-NOT: air.refeed_count
air.channel @c7 [1, 1]
func.func @two_puts(%m: memref<64xi32, 2>, %n: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.put @c7[] (%m[] [] []) : (memref<64xi32, 2>)
    air.channel.put @c7[] (%n[] [] []) : (memref<64xi32, 2>)
  }
  return
}

// -----

// Not a re-broadcast: the body rewrites the source between sends, so the loop
// really is N productions. Left unchanged.

// CHECK-LABEL: @rewritten_between_sends
// CHECK: scf.for
// CHECK: func.call @norm
// CHECK-NOT: air.refeed_count
air.channel @c8 [1, 1]
func.func private @norm(memref<64xi32, 2>)
func.func @rewritten_between_sends(%m: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c3 step %c1 {
    func.call @norm(%m) : (memref<64xi32, 2>) -> ()
    air.channel.put @c8[] (%m[] [] []) : (memref<64xi32, 2>)
  }
  return
}

// -----

// A dynamic trip count cannot become a lock init. Left unchanged.

// CHECK-LABEL: @dynamic_trip
// CHECK: scf.for
// CHECK-NOT: air.refeed_count
air.channel @c9 [1, 1]
func.func @dynamic_trip(%m: memref<64xi32, 2>, %ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    air.channel.put @c9[] (%m[] [] []) : (memref<64xi32, 2>)
  }
  return
}

// -----

// An async put carrying a dependency other than the loop's token would lose
// that edge if hoisted. Left unchanged.

// CHECK-LABEL: @extra_async_dep
// CHECK: scf.for
// CHECK-NOT: air.refeed_count
air.channel @c10 [1, 1]
func.func @extra_async_dep() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %t, %buf = air.execute -> (memref<64xi32, 2>) {
    %a = memref.alloc() : memref<64xi32, 2>
    air.execute_terminator %a : memref<64xi32, 2>
  }
  %u = air.wait_all async
  %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %p = air.channel.put async [%tok, %u] @c10[] (%buf[] [] []) : (memref<64xi32, 2>)
    scf.yield %p : !air.async.token
  }
  return
}

// -----

// A trip count of one is a plain put: the loop goes, no count is recorded.

// CHECK-LABEL: @single_trip
// CHECK-NOT: scf.for
// CHECK: air.channel.put
// CHECK-NOT: air.refeed_count
air.channel @c11 [1, 1]
func.func @single_trip(%m: memref<64xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    air.channel.put @c11[] (%m[] [] []) : (memref<64xi32, 2>)
  }
  return
}

// -----

// An L2 source whose memref.alloc is not recoverable (here a block argument)
// has no carrier the memtile lock allocator reads. Writing the count on the put
// would be silently ineffective and deadlock on device, so it is an error.

air.channel @c12 [1, 1]
func.func @l2_no_alloc(%m: memref<64xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    // expected-error @+1 {{air.refeed_count has no carrier the memtile lock allocator reads}}
    air.channel.put @c12[] (%m[] [] []) : (memref<64xi32, 1>)
  }
  return
}
