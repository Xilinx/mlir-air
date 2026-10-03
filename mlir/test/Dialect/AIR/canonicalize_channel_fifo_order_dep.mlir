//===- canonicalize_channel_fifo_order_dep.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize -split-input-file | FileCheck %s

// Channel endpoint ordering edges are resource dependencies: they exist because
// the two ops share a FIFO, not a buffer. Dead-edge removal keeps an edge when
// the two ops share a resource, and `chan_name` is a symbol reference, so
// same-channel edges survive.
//
// These tests pin that. Were it otherwise, ordering edges would evaporate at
// every canonicalize and could only be established after the last one.

// -----

// Two same-slot gets on different buffers: no memref dependency relates them,
// so only the shared channel symbol keeps the edge alive.

// CHECK-LABEL: func.func @fifo_dep_direct
// CHECK: %[[G0:.*]] = air.channel.get async {{.*}}@channel_0
// CHECK: air.channel.get async [{{.*}}%[[G0]]{{.*}}] @channel_0

module {
  air.channel @channel_0 [1]
  func.func @fifo_dep_direct() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @channel_0[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1, %g0] @channel_0[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// Control for the case above: identical shape, but the two ops sit on different
// channels. No shared resource, no memref dependency, so the edge is removed.

// CHECK-LABEL: func.func @fifo_dep_cross_channel_dropped
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_1
// CHECK-NOT: air.channel.get async [%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}] @channel_2
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_2

module {
  air.channel @channel_1 [1]
  air.channel @channel_2 [1]
  func.func @fifo_dep_cross_channel_dropped() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @channel_1[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1, %g0] @channel_2[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// The iter_arg is the only thing sequencing iteration i+1's get after
// iteration i's. Folding it away would lose the cross-iteration order; neither
// iter-arg folding rule applies, since the arg is used and the yielded value
// differs from it.

// CHECK-LABEL: func.func @fifo_dep_loop_carried
// CHECK: scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (!air.async.token)
// CHECK: %[[G:.*]] = air.channel.get async [%[[IT]]] @channel_3
// CHECK: scf.yield %[[G]] : !air.async.token

module {
  air.channel @channel_3 [1]
  func.func @fifo_dep_loop_carried() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %l = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g = air.channel.get async [%it] @channel_3[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Two endpoints per body: the intra-body edge (%g1 after %g0) and the
// cross-iteration edge through the iter_arg must both survive.

// CHECK-LABEL: func.func @fifo_dep_loop_carried_two_endpoints
// CHECK: scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (!air.async.token)
// CHECK: %[[G0:.*]] = air.channel.get async [%[[IT]]] @channel_4
// CHECK: %[[G1:.*]] = air.channel.get async [{{.*}}%[[G0]]{{.*}}] @channel_4
// CHECK: scf.yield %[[G1]] : !air.async.token

module {
  air.channel @channel_4 [1]
  func.func @fifo_dep_loop_carried_two_endpoints() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %l = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%it] @channel_4[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          %g1 = air.channel.get async [%t1, %g0] @channel_4[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Ordering threaded out through an affine.if token result, as emitted for
// endpoints in sibling conditionals. The edge names the affine.if result, not
// the guarded op's own token.

// CHECK-LABEL: func.func @fifo_dep_through_affine_if
// CHECK: %[[IF0:.*]] = affine.if
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_5
// CHECK: affine.if
// CHECK: air.channel.get async [{{.*}}%[[IF0]]{{.*}}] @channel_5

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_5 [1, 1] {broadcast_shape = [1, 2]}
  func.func @fifo_dep_through_affine_if() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c2_0) {
          %t0, %r0 = air.execute -> (memref<8xi32, 2 : i32>) {
            %m = memref.alloc() : memref<8xi32, 2 : i32>
            air.execute_terminator %m : memref<8xi32, 2 : i32>
          }
          %t1, %r1 = air.execute -> (memref<8xi32, 2 : i32>) {
            %m = memref.alloc() : memref<8xi32, 2 : i32>
            air.execute_terminator %m : memref<8xi32, 2 : i32>
          }
          %w = air.wait_all async
          %i0 = affine.if #set()[%x, %y] -> !air.async.token {
            %g0 = air.channel.get async [%t0] @channel_5[%x, %y] (%r0[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g0 : !air.async.token
          } else {
            affine.yield %w : !air.async.token
          }
          %i1 = affine.if #set()[%x, %y] -> !air.async.token {
            %g1 = air.channel.get async [%t1, %i0] @channel_5[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g1 : !air.async.token
          } else {
            affine.yield %w : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// A transitive path is not durable. The two @channel_6 gets are ordered only
// through %p, a put on @channel_7; both links relate ops sharing neither a
// channel nor a buffer, so both are false dependencies and both are removed.
// %p is left with no dependencies and the two gets are left unordered.
//
// Hence ordering must be emitted as a direct same-slot edge, and must be
// re-established after any rewrite that leaves only a transitive path.

// CHECK-LABEL: func.func @fifo_dep_transitive_is_fragile
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_6
// CHECK-NOT: air.channel.put async [{{.*}}] @channel_7
// CHECK: air.channel.put async {{.*}}@channel_7
// CHECK-NOT: air.channel.get async [%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}] @channel_6
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_6

module {
  air.channel @channel_6 [1]
  air.channel @channel_7 [1]
  func.func @fifo_dep_transitive_is_fragile() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t2, %r2 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @channel_6[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %p = air.channel.put async [%g0] @channel_7[%c0] (%r2[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1, %p] @channel_6[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}
