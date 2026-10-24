//===- air_scf_parallel_in_loop.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s

// An scf.parallel inside an scf.for in a segment. The launch and the segment
// become scf.parallel ops too, and each scf.parallel becomes a nest of
// affine.for loops holding a copy of its body. The inner one is converted
// first; the launch's and the segment's copies of their bodies then hold only
// its loops, so @work is called once per iteration.

// CHECK-LABEL: func.func @par_in_for
// CHECK: scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NEXT: func.call @work(%[[K]], %[[I]])
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NOT: scf.parallel
// CHECK-NOT: func.call @work
// CHECK: return

func.func private @work(index, index)
func.func @par_in_for() {
  %c2 = arith.constant 2 : index
  air.launch (%x, %y) in (%sx=%c2, %sy=%c2) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      scf.for %k = %c0 to %c3 step %c1 {
        scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
          func.call @work(%k, %i) : (index, index) -> ()
          scf.reduce
        }
      }
    }
  }
  return
}

// An scf.parallel whose token an enclosing scf.for passes on. The loop's
// yield is a terminator and must still pass a value, so it gets an event that
// is already complete in place of the parallel's.

// CHECK-LABEL: func.func @par_token_in_for
// CHECK: %[[R:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!airrt.event) {
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NEXT: func.call @work(%[[K]], %[[I]])
// CHECK: }
// CHECK-NEXT: %[[DONE:.*]] = airrt.wait_all : !airrt.event
// CHECK-NEXT: scf.yield %[[DONE]] : !airrt.event
// CHECK: airrt.wait_all %[[R]]
// CHECK-NOT: scf.parallel

func.func @par_token_in_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %t0 = air.wait_all async
  %r = scf.for %k = %c0 to %c3 step %c1 iter_args(%it = %t0) -> (!air.async.token) {
    %p = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%it) -> !air.async.token {
      %e = air.execute {
        func.call @work(%k, %i) : (index, index) -> ()
      }
      scf.reduce(%e : !air.async.token) {
      ^bb0(%a: !air.async.token, %b: !air.async.token):
        %w = air.wait_all async [%a, %b]
        scf.reduce.return %w : !air.async.token
      }
    }
    scf.yield %p : !air.async.token
  }
  air.wait_all [%r]
  return
}

// Nested scf.parallel ops with tokens. The inner one is converted first; its
// token, taken by the outer one's scf.reduce, is replaced by a complete event,
// and the outer one's body is then copied once into its loop. The wait on the
// outer token loses that operand.

// CHECK-LABEL: func.func @nested_par_tokens
// CHECK: affine.for %[[J:.*]] = 0 to 2 {
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NEXT: func.call @work(%[[J]], %[[I]])
// CHECK-NOT: func.call @work
// CHECK: airrt.wait_all{{$}}
// CHECK-NOT: scf.parallel
// CHECK: return

func.func @nested_par_tokens() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %t0 = air.wait_all async
  %p = scf.parallel (%j) = (%c0) to (%c2) step (%c1) init (%t0) -> !air.async.token {
    %q = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%t0) -> !air.async.token {
      %e = air.execute {
        func.call @work(%j, %i) : (index, index) -> ()
      }
      scf.reduce(%e : !air.async.token) {
      ^bb0(%a: !air.async.token, %b: !air.async.token):
        %w = air.wait_all async [%a, %b]
        scf.reduce.return %w : !air.async.token
      }
    }
    scf.reduce(%q : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  air.wait_all [%p]
  return
}
