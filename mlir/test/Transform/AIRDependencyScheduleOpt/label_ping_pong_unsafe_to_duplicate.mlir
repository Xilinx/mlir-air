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

// -----

// Labeling a loop unrolls its body by 2, which duplicates a nested loop and
// whatever that loop allocates per trip. So an unsafe inner loop disqualifies
// the outer candidate as well. Rejecting only the inner one would instead
// release the outer, which the deepest-qualifying-loop rule had been
// suppressing, and duplicate a bigger buffer at a worse level.

// CHECK-LABEL: @nested_unsafe_disqualifies_outer
// CHECK-NOT: hoist_alloc
// CHECK-NOT: unroll
air.channel @c4 [1, 1]
air.channel @c5 [1, 1]
func.func private @accum(memref<64xbf16, 2>, memref<16xf32, 2>)
func.func @nested_unsafe_disqualifies_outer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %t = air.wait_all async
  %outer = scf.for %i = %c0 to %c4 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %tok0, %big = air.execute [%tok] -> (memref<4096xbf16, 2>) {
      %alloc = memref.alloc() : memref<4096xbf16, 2>
      air.execute_terminator %alloc : memref<4096xbf16, 2>
    }
    %g = air.channel.get async [%tok0] @c4[] (%big[] [] []) : (memref<4096xbf16, 2>)
    %inner = scf.for %j = %c0 to %c8 step %c1 iter_args(%itok = %g) -> (!air.async.token) {
      %tok1, %buf = air.execute [%itok] -> (memref<64xbf16, 2>) {
        %alloc = memref.alloc() : memref<64xbf16, 2>
        air.execute_terminator %alloc : memref<64xbf16, 2>
      }
      %tok2, %acc = air.execute -> (memref<16xf32, 2>) {
        %alloc = memref.alloc() : memref<16xf32, 2>
        air.execute_terminator %alloc : memref<16xf32, 2>
      }
      %g2 = air.channel.get async [%tok1] @c5[] (%buf[] [] []) : (memref<64xbf16, 2>)
      %tok3 = air.execute [%g2, %tok2] {
        func.call @accum(%buf, %acc) : (memref<64xbf16, 2>, memref<16xf32, 2>) -> ()
      }
      %tok4 = air.execute [%tok3] {
        memref.dealloc %buf : memref<64xbf16, 2>
      }
      %tok5 = air.execute [%tok3] {
        memref.dealloc %acc : memref<16xf32, 2>
      }
      scf.yield %tok4 : !air.async.token
    }
    %tok6 = air.execute [%inner] {
      memref.dealloc %big : memref<4096xbf16, 2>
    }
    scf.yield %tok6 : !air.async.token
  }
  return
}

// -----

// The callee is defined in the module (a loop nest outlined from the herd) and
// only writes the per-iteration buffer, so its call is the buffer's definite
// first write: the loop is double buffered as with a channel.get.

// CHECK-LABEL: @defined_callee_writes_only
// CHECK: memref.alloc() {hoist_alloc = true} : memref<16xf32, 2>
// CHECK: } {unroll = 2 : i32}
air.channel @c1 [1, 1]
func.func private @fill_tile(%src: memref<64xbf16, 2>, %dst: memref<16xf32, 2>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : bf16
  %v = vector.transfer_read %src[%c0], %p {in_bounds = [true]} : memref<64xbf16, 2>, vector<16xbf16>
  %f = arith.extf %v : vector<16xbf16> to vector<16xf32>
  %view = memref.subview %dst[0] [16] [1] : memref<16xf32, 2> to memref<16xf32, strided<[1]>, 2>
  vector.transfer_write %f, %view[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, strided<[1]>, 2>
  return
}
func.func @defined_callee_writes_only() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t = air.wait_all async
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
      %alloc = memref.alloc() : memref<64xbf16, 2>
      air.execute_terminator %alloc : memref<64xbf16, 2>
    }
    %tok1, %out = air.execute -> (memref<16xf32, 2>) {
      %alloc = memref.alloc() : memref<16xf32, 2>
      air.execute_terminator %alloc : memref<16xf32, 2>
    }
    %g = air.channel.get async [%tok0] @c1[] (%buf[] [] []) : (memref<64xbf16, 2>)
    %tok2 = air.execute [%g, %tok1] {
      func.call @fill_tile(%buf, %out) : (memref<64xbf16, 2>, memref<16xf32, 2>) -> ()
    }
    %tok3 = air.execute [%tok2] {
      memref.dealloc %buf : memref<64xbf16, 2>
    }
    %tok4 = air.execute [%tok2] {
      memref.dealloc %out : memref<16xf32, 2>
    }
    scf.yield %tok3 : !air.async.token
  }
  return
}

// -----

// The same defined callee, but it also reads the buffer before writing it:
// the value may flow from the previous iteration, so the loop stays single.

// CHECK-LABEL: @defined_callee_reads
// CHECK-NOT: unroll
air.channel @c1 [1, 1]
func.func private @acc_tile(%src: memref<64xbf16, 2>, %dst: memref<16xf32, 2>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32
  %a = vector.transfer_read %dst[%c0], %p {in_bounds = [true]} : memref<16xf32, 2>, vector<16xf32>
  %s = arith.addf %a, %a : vector<16xf32>
  vector.transfer_write %s, %dst[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
  return
}
func.func @defined_callee_reads() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t = air.wait_all async
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %tok0, %buf = air.execute [%tok] -> (memref<64xbf16, 2>) {
      %alloc = memref.alloc() : memref<64xbf16, 2>
      air.execute_terminator %alloc : memref<64xbf16, 2>
    }
    %tok1, %out = air.execute -> (memref<16xf32, 2>) {
      %alloc = memref.alloc() : memref<16xf32, 2>
      air.execute_terminator %alloc : memref<16xf32, 2>
    }
    %g = air.channel.get async [%tok0] @c1[] (%buf[] [] []) : (memref<64xbf16, 2>)
    %tok2 = air.execute [%g, %tok1] {
      func.call @acc_tile(%buf, %out) : (memref<64xbf16, 2>, memref<16xf32, 2>) -> ()
    }
    %tok3 = air.execute [%tok2] {
      memref.dealloc %buf : memref<64xbf16, 2>
    }
    %tok4 = air.execute [%tok2] {
      memref.dealloc %out : memref<16xf32, 2>
    }
    scf.yield %tok3 : !air.async.token
  }
  return
}
