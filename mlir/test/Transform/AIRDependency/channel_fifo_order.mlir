//===- channel_fifo_order.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-enforce-channel-fifo-order | FileCheck %s

// A channel is an ordered FIFO, so two gets on the same channel + index +
// direction must be serialized even when they touch different buffers. The
// pass adds a direct dependency from each op to the nearest preceding match in
// the same sequential scope, looking through conditional regions but not
// through loops.

// CHECK-LABEL: func.func @chan_fifo_get
// CHECK: %[[G0:.*]] = air.channel.get async {{.*}}@channel_0
// CHECK: air.channel.get async [{{.*}}%[[G0]]] {{.*}}@channel_0

module {
  air.channel @channel_0 [1]
  func.func @chan_fifo_get() {
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
        %g1 = air.channel.get async [%t1] @channel_0[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// Different indices => different FIFO sub-channels => left independent. Each get
// keeps a single-token dependency list (its original token only); the CHECK-NOT
// asserts no serializing edge to the sibling get was added.

// CHECK-LABEL: func.func @chan_diff_index
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_1[%c0
// CHECK-NOT: air.channel.get async [%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}] @channel_1
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_1[%c1

module {
  air.channel @channel_1 [2]
  func.func @chan_diff_index() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @channel_1[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1] @channel_1[%c1_0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// Same-index puts on one channel are serialized in program order, exactly like
// gets (the FIFO order applies per direction).

// CHECK-LABEL: func.func @chan_fifo_put
// CHECK: %[[P0:.*]] = air.channel.put async {{.*}}@channel_2
// CHECK: air.channel.put async [{{.*}}%[[P0]]] {{.*}}@channel_2

module {
  air.channel @channel_2 [1]
  func.func @chan_fifo_put() {
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
        %p0 = air.channel.put async [%t0] @channel_2[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %p1 = air.channel.put async [%t1] @channel_2[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// A put and a get on the same channel + same index are opposite directions and
// are NOT serialized against each other. The get keeps its single-token list.

// CHECK-LABEL: func.func @chan_put_get_indep
// CHECK: air.channel.put async [%{{[a-z_0-9]+}}] @channel_3
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_3

module {
  air.channel @channel_3 [1]
  func.func @chan_put_get_indep() {
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
        %p0 = air.channel.put async [%t0] @channel_3[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g0 = air.channel.get async [%t1] @channel_3[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// Three same-index gets form a nearest-preceding chain: g1 depends on g0 and g2
// depends on g1. This exercises the transitive ordering the pass relies on.

// CHECK-LABEL: func.func @chan_fifo_chain
// CHECK: %[[C0:.*]] = air.channel.get async {{.*}}@channel_4
// CHECK: %[[C1:.*]] = air.channel.get async [{{.*}}%[[C0]]] {{.*}}@channel_4
// CHECK: air.channel.get async [{{.*}}%[[C1]]] {{.*}}@channel_4

module {
  air.channel @channel_4 [1]
  func.func @chan_fifo_chain() {
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
        %g0 = air.channel.get async [%t0] @channel_4[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1] @channel_4[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
        %g2 = air.channel.get async [%t2] @channel_4[%c0] (%r2[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----

// Broadcast lowering wraps each endpoint in an affine.if guard, putting the two
// gets in separate blocks while leaving them in one sequential scope. The
// ordering edge names the first guard's token result.
//
// Keying the scan on Block* skips these endpoints entirely, so they race.

// CHECK-LABEL: func.func @chan_fifo_through_affine_if
// CHECK: %[[IF0:.*]] = affine.if
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_5
// CHECK: affine.if
// CHECK: air.channel.get async [{{.*}}%[[IF0]]{{.*}}] @channel_5

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_5 [1, 1] {broadcast_shape = [1, 2]}
  func.func @chan_fifo_through_affine_if() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
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
            %g1 = air.channel.get async [%t1] @channel_5[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
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

// Endpoints in the then- and else-branch of one conditional are mutually
// exclusive. Nothing to order, and naming the enclosing affine.if's own result
// would be a self-dependency, so neither get gains an edge.

// CHECK-LABEL: func.func @chan_fifo_exclusive_branches
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_6
// CHECK-NOT: air.channel.get async [%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}] @channel_6
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_6

#set1 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_6 [1, 1] {broadcast_shape = [1, 2]}
  func.func @chan_fifo_exclusive_branches() {
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
          %i0 = affine.if #set1()[%x, %y] -> !air.async.token {
            %g0 = air.channel.get async [%t0] @channel_6[%x, %y] (%r0[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g0 : !air.async.token
          } else {
            %g1 = air.channel.get async [%t1] @channel_6[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g1 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Gets in two loops one after the other: the second loop's get waits for the
// first loop, and for its own previous iteration through the iteration
// argument.

// CHECK-LABEL: func.func @chan_fifo_distinct_loops
// CHECK: %[[L0:.*]] = scf.for
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_7
// CHECK: scf.for {{.*}} iter_args(%[[A:.*]] = %{{[a-z_0-9]+}})
// CHECK: air.channel.get async [%[[L0]], %[[A]]] @channel_7

module {
  air.channel @channel_7 [1]
  func.func @chan_fifo_distinct_loops() {
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
        %l0 = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%it] @channel_7[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g0 : !air.async.token
        }
        %l1 = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%it = %t1) -> (!air.async.token) {
          %g1 = air.channel.get async [%it] @channel_7[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Endpoints under unrelated conditions are not ordered. On the arm that does not
// run, a conditional's token result carries whatever the other arm yields, which
// says nothing about the endpoint inside -- so an edge through it would be
// fictional, and would keep the other arm's token live for no reason.

// CHECK-LABEL: func.func @chan_fifo_distinct_guards
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_8
// CHECK-NOT: air.channel.get async [%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}] @channel_8
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_8

module {
  air.channel @channel_8 [1]
  func.func @chan_fifo_distinct_guards(%p0: i1, %p1: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%lp0=%p0, %lp1=%p1) : i1, i1 {
      air.segment @seg args(%q0=%lp0, %q1=%lp1) : i1, i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %i0 = scf.if %q0 -> (!air.async.token) {
          %g0 = air.channel.get async [%t0] @channel_8[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g0 : !air.async.token
        } else {
          scf.yield %w : !air.async.token
        }
        %i1 = scf.if %q1 -> (!air.async.token) {
          %g1 = air.channel.get async [%t1] @channel_8[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        } else {
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// scf.index_switch arms are conditional scopes like scf.if arms. Endpoints in
// the same arm of two switches on the same value are ordered against each other.

// CHECK-LABEL: func.func @chan_fifo_index_switch
// CHECK: %[[S0:.*]] = scf.index_switch
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_9
// CHECK: scf.index_switch
// CHECK: air.channel.get async [{{.*}}%[[S0]]{{.*}}] @channel_9

module {
  air.channel @channel_9 [1]
  func.func @chan_fifo_index_switch(%arg: index) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%larg=%arg) : index {
      air.segment @seg args(%sel=%larg) : index {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %s0 = scf.index_switch %sel -> !air.async.token
        case 0 {
          %g0 = air.channel.get async [%t0] @channel_9[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g0 : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
        %s1 = scf.index_switch %sel -> !air.async.token
        case 0 {
          %g1 = air.channel.get async [%t1] @channel_9[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// A conditional can carry several token results, and the one this endpoint
// yields need not be the first: rebuildIndexSwitchWithTrailingAsyncToken
// appends its summary token after the existing results. Promotion has to pick
// the result the arm yields this endpoint into -- here result #1, not #0 --
// or the published token says nothing about the endpoint.

// CHECK-LABEL: func.func @chan_fifo_second_token_result
// CHECK: %[[IF0:.*]]:2 = scf.if
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_10
// CHECK: scf.if
// CHECK: air.channel.get async [{{.*}}%[[IF0]]#1{{.*}}] @channel_10

module {
  air.channel @channel_10 [1]
  func.func @chan_fifo_second_token_result(%p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%lp=%p) : i1 {
      air.segment @seg args(%q=%lp) : i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %i0:2 = scf.if %q -> (!air.async.token, !air.async.token) {
          %u = air.wait_all async
          %g0 = air.channel.get async [%t0] @channel_10[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %u, %g0 : !air.async.token, !air.async.token
        } else {
          scf.yield %w, %w : !air.async.token, !air.async.token
        }
        %i1:2 = scf.if %q -> (!air.async.token, !air.async.token) {
          %u = air.wait_all async
          %g1 = air.channel.get async [%t1] @channel_10[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %u, %g1 : !air.async.token, !air.async.token
        } else {
          scf.yield %w, %w : !air.async.token, !air.async.token
        }
      }
    }
    return
  }
}

// -----

// The case yields a wait on a branch result that carries the get's token in one
// arm only. That result does not say the get finished, so the next switch's get
// is not ordered behind it.

// CHECK-LABEL: func.func @chan_fifo_one_arm_wait
// CHECK: scf.index_switch
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_10
// CHECK: scf.index_switch
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_10

module {
  air.channel @channel_10 [1]
  func.func @chan_fifo_one_arm_wait(%arg: index, %p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%larg=%arg, %lp=%p) : index, i1 {
      air.segment @seg args(%sel=%larg, %go=%lp) : index, i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %s0 = scf.index_switch %sel -> !air.async.token
        case 0 {
          %g0 = air.channel.get async [%t0] @channel_10[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          %c = scf.if %go -> (!air.async.token) {
            scf.yield %g0 : !air.async.token
          } else {
            scf.yield %w : !air.async.token
          }
          %j = air.wait_all async [%c]
          scf.yield %j : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
        %s1 = scf.index_switch %sel -> !air.async.token
        case 0 {
          %g1 = air.channel.get async [%t1] @channel_10[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// The same, with every arm waiting on the get: the next get follows the switch.

// CHECK-LABEL: func.func @chan_fifo_all_arms_wait
// CHECK: %[[S0:.*]] = scf.index_switch
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_11
// CHECK: scf.index_switch
// CHECK: air.channel.get async [{{.*}}%[[S0]]{{.*}}] @channel_11

module {
  air.channel @channel_11 [1]
  func.func @chan_fifo_all_arms_wait(%arg: index, %p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%larg=%arg, %lp=%p) : index, i1 {
      air.segment @seg args(%sel=%larg, %go=%lp) : index, i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %s0 = scf.index_switch %sel -> !air.async.token
        case 0 {
          %g0 = air.channel.get async [%t0] @channel_11[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          %c = scf.if %go -> (!air.async.token) {
            scf.yield %g0 : !air.async.token
          } else {
            %e = air.wait_all async [%g0]
            scf.yield %e : !air.async.token
          }
          %j = air.wait_all async [%c]
          scf.yield %j : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
        %s1 = scf.index_switch %sel -> !air.async.token
        case 0 {
          %g1 = air.channel.get async [%t1] @channel_11[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Two switches on selectors computed the same way, 30 additions deep: the same
// condition, found without comparing the shared subexpressions repeatedly.

// CHECK-LABEL: func.func @chan_fifo_deep_selector
// CHECK: %[[S0:.*]] = scf.index_switch
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_12
// CHECK: scf.index_switch
// CHECK: air.channel.get async [{{.*}}%[[S0]]{{.*}}] @channel_12

module {
  air.channel @channel_12 [1]
  func.func @chan_fifo_deep_selector(%arg: index, %p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%larg=%arg, %lp=%p) : index, i1 {
      air.segment @seg args(%sel=%larg, %go=%lp) : index, i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %x0 = arith.addi %sel, %sel : index
        %x1 = arith.addi %x0, %x0 : index
        %x2 = arith.addi %x1, %x1 : index
        %x3 = arith.addi %x2, %x2 : index
        %x4 = arith.addi %x3, %x3 : index
        %x5 = arith.addi %x4, %x4 : index
        %x6 = arith.addi %x5, %x5 : index
        %x7 = arith.addi %x6, %x6 : index
        %x8 = arith.addi %x7, %x7 : index
        %x9 = arith.addi %x8, %x8 : index
        %x10 = arith.addi %x9, %x9 : index
        %x11 = arith.addi %x10, %x10 : index
        %x12 = arith.addi %x11, %x11 : index
        %x13 = arith.addi %x12, %x12 : index
        %x14 = arith.addi %x13, %x13 : index
        %x15 = arith.addi %x14, %x14 : index
        %x16 = arith.addi %x15, %x15 : index
        %x17 = arith.addi %x16, %x16 : index
        %x18 = arith.addi %x17, %x17 : index
        %x19 = arith.addi %x18, %x18 : index
        %x20 = arith.addi %x19, %x19 : index
        %x21 = arith.addi %x20, %x20 : index
        %x22 = arith.addi %x21, %x21 : index
        %x23 = arith.addi %x22, %x22 : index
        %x24 = arith.addi %x23, %x23 : index
        %x25 = arith.addi %x24, %x24 : index
        %x26 = arith.addi %x25, %x25 : index
        %x27 = arith.addi %x26, %x26 : index
        %x28 = arith.addi %x27, %x27 : index
        %x29 = arith.addi %x28, %x28 : index
        %y0 = arith.addi %sel, %sel : index
        %y1 = arith.addi %y0, %y0 : index
        %y2 = arith.addi %y1, %y1 : index
        %y3 = arith.addi %y2, %y2 : index
        %y4 = arith.addi %y3, %y3 : index
        %y5 = arith.addi %y4, %y4 : index
        %y6 = arith.addi %y5, %y5 : index
        %y7 = arith.addi %y6, %y6 : index
        %y8 = arith.addi %y7, %y7 : index
        %y9 = arith.addi %y8, %y8 : index
        %y10 = arith.addi %y9, %y9 : index
        %y11 = arith.addi %y10, %y10 : index
        %y12 = arith.addi %y11, %y11 : index
        %y13 = arith.addi %y12, %y12 : index
        %y14 = arith.addi %y13, %y13 : index
        %y15 = arith.addi %y14, %y14 : index
        %y16 = arith.addi %y15, %y15 : index
        %y17 = arith.addi %y16, %y16 : index
        %y18 = arith.addi %y17, %y17 : index
        %y19 = arith.addi %y18, %y18 : index
        %y20 = arith.addi %y19, %y19 : index
        %y21 = arith.addi %y20, %y20 : index
        %y22 = arith.addi %y21, %y21 : index
        %y23 = arith.addi %y22, %y22 : index
        %y24 = arith.addi %y23, %y23 : index
        %y25 = arith.addi %y24, %y24 : index
        %y26 = arith.addi %y25, %y25 : index
        %y27 = arith.addi %y26, %y26 : index
        %y28 = arith.addi %y27, %y27 : index
        %y29 = arith.addi %y28, %y28 : index
        %s0 = scf.index_switch %x29 -> !air.async.token
        case 0 {
          %g0 = air.channel.get async [%t0] @channel_12[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          %c = scf.if %go -> (!air.async.token) {
            scf.yield %g0 : !air.async.token
          } else {
            %e = air.wait_all async [%g0]
            scf.yield %e : !air.async.token
          }
          %j = air.wait_all async [%c]
          scf.yield %j : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
        %s1 = scf.index_switch %y29 -> !air.async.token
        case 0 {
          %g1 = air.channel.get async [%t1] @channel_12[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
        default {
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// A get in a loop, then 40 branches whose arms all wait on the one before:
// the loop's result waits on the get, so the get after the loop follows it.
// Each branch result is decided once, not once per path through the chain.

// CHECK-LABEL: func.func @chan_fifo_long_branch_chain
// CHECK: %[[L:.*]] = scf.for
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_13
// CHECK: air.channel.get async [%[[L]]] @channel_13

module {
  air.channel @channel_13 [1]
  func.func @chan_fifo_long_branch_chain(%p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%lp=%p) : i1 {
      air.segment @seg args(%go=%lp) : i1 {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1s = arith.constant 1 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %l = scf.for %i = %c0 to %c2 step %c1s iter_args(%t = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%t] @channel_13[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          %s1 = scf.if %go -> (!air.async.token) {
            %w1 = air.wait_all async [%g0]
            scf.yield %w1 : !air.async.token
          } else {
            %e1 = air.wait_all async [%g0]
            scf.yield %e1 : !air.async.token
          }
          %s2 = scf.if %go -> (!air.async.token) {
            %w2 = air.wait_all async [%s1]
            scf.yield %w2 : !air.async.token
          } else {
            %e2 = air.wait_all async [%s1]
            scf.yield %e2 : !air.async.token
          }
          %s3 = scf.if %go -> (!air.async.token) {
            %w3 = air.wait_all async [%s2]
            scf.yield %w3 : !air.async.token
          } else {
            %e3 = air.wait_all async [%s2]
            scf.yield %e3 : !air.async.token
          }
          %s4 = scf.if %go -> (!air.async.token) {
            %w4 = air.wait_all async [%s3]
            scf.yield %w4 : !air.async.token
          } else {
            %e4 = air.wait_all async [%s3]
            scf.yield %e4 : !air.async.token
          }
          %s5 = scf.if %go -> (!air.async.token) {
            %w5 = air.wait_all async [%s4]
            scf.yield %w5 : !air.async.token
          } else {
            %e5 = air.wait_all async [%s4]
            scf.yield %e5 : !air.async.token
          }
          %s6 = scf.if %go -> (!air.async.token) {
            %w6 = air.wait_all async [%s5]
            scf.yield %w6 : !air.async.token
          } else {
            %e6 = air.wait_all async [%s5]
            scf.yield %e6 : !air.async.token
          }
          %s7 = scf.if %go -> (!air.async.token) {
            %w7 = air.wait_all async [%s6]
            scf.yield %w7 : !air.async.token
          } else {
            %e7 = air.wait_all async [%s6]
            scf.yield %e7 : !air.async.token
          }
          %s8 = scf.if %go -> (!air.async.token) {
            %w8 = air.wait_all async [%s7]
            scf.yield %w8 : !air.async.token
          } else {
            %e8 = air.wait_all async [%s7]
            scf.yield %e8 : !air.async.token
          }
          %s9 = scf.if %go -> (!air.async.token) {
            %w9 = air.wait_all async [%s8]
            scf.yield %w9 : !air.async.token
          } else {
            %e9 = air.wait_all async [%s8]
            scf.yield %e9 : !air.async.token
          }
          %s10 = scf.if %go -> (!air.async.token) {
            %w10 = air.wait_all async [%s9]
            scf.yield %w10 : !air.async.token
          } else {
            %e10 = air.wait_all async [%s9]
            scf.yield %e10 : !air.async.token
          }
          %s11 = scf.if %go -> (!air.async.token) {
            %w11 = air.wait_all async [%s10]
            scf.yield %w11 : !air.async.token
          } else {
            %e11 = air.wait_all async [%s10]
            scf.yield %e11 : !air.async.token
          }
          %s12 = scf.if %go -> (!air.async.token) {
            %w12 = air.wait_all async [%s11]
            scf.yield %w12 : !air.async.token
          } else {
            %e12 = air.wait_all async [%s11]
            scf.yield %e12 : !air.async.token
          }
          %s13 = scf.if %go -> (!air.async.token) {
            %w13 = air.wait_all async [%s12]
            scf.yield %w13 : !air.async.token
          } else {
            %e13 = air.wait_all async [%s12]
            scf.yield %e13 : !air.async.token
          }
          %s14 = scf.if %go -> (!air.async.token) {
            %w14 = air.wait_all async [%s13]
            scf.yield %w14 : !air.async.token
          } else {
            %e14 = air.wait_all async [%s13]
            scf.yield %e14 : !air.async.token
          }
          %s15 = scf.if %go -> (!air.async.token) {
            %w15 = air.wait_all async [%s14]
            scf.yield %w15 : !air.async.token
          } else {
            %e15 = air.wait_all async [%s14]
            scf.yield %e15 : !air.async.token
          }
          %s16 = scf.if %go -> (!air.async.token) {
            %w16 = air.wait_all async [%s15]
            scf.yield %w16 : !air.async.token
          } else {
            %e16 = air.wait_all async [%s15]
            scf.yield %e16 : !air.async.token
          }
          %s17 = scf.if %go -> (!air.async.token) {
            %w17 = air.wait_all async [%s16]
            scf.yield %w17 : !air.async.token
          } else {
            %e17 = air.wait_all async [%s16]
            scf.yield %e17 : !air.async.token
          }
          %s18 = scf.if %go -> (!air.async.token) {
            %w18 = air.wait_all async [%s17]
            scf.yield %w18 : !air.async.token
          } else {
            %e18 = air.wait_all async [%s17]
            scf.yield %e18 : !air.async.token
          }
          %s19 = scf.if %go -> (!air.async.token) {
            %w19 = air.wait_all async [%s18]
            scf.yield %w19 : !air.async.token
          } else {
            %e19 = air.wait_all async [%s18]
            scf.yield %e19 : !air.async.token
          }
          %s20 = scf.if %go -> (!air.async.token) {
            %w20 = air.wait_all async [%s19]
            scf.yield %w20 : !air.async.token
          } else {
            %e20 = air.wait_all async [%s19]
            scf.yield %e20 : !air.async.token
          }
          %s21 = scf.if %go -> (!air.async.token) {
            %w21 = air.wait_all async [%s20]
            scf.yield %w21 : !air.async.token
          } else {
            %e21 = air.wait_all async [%s20]
            scf.yield %e21 : !air.async.token
          }
          %s22 = scf.if %go -> (!air.async.token) {
            %w22 = air.wait_all async [%s21]
            scf.yield %w22 : !air.async.token
          } else {
            %e22 = air.wait_all async [%s21]
            scf.yield %e22 : !air.async.token
          }
          %s23 = scf.if %go -> (!air.async.token) {
            %w23 = air.wait_all async [%s22]
            scf.yield %w23 : !air.async.token
          } else {
            %e23 = air.wait_all async [%s22]
            scf.yield %e23 : !air.async.token
          }
          %s24 = scf.if %go -> (!air.async.token) {
            %w24 = air.wait_all async [%s23]
            scf.yield %w24 : !air.async.token
          } else {
            %e24 = air.wait_all async [%s23]
            scf.yield %e24 : !air.async.token
          }
          %s25 = scf.if %go -> (!air.async.token) {
            %w25 = air.wait_all async [%s24]
            scf.yield %w25 : !air.async.token
          } else {
            %e25 = air.wait_all async [%s24]
            scf.yield %e25 : !air.async.token
          }
          %s26 = scf.if %go -> (!air.async.token) {
            %w26 = air.wait_all async [%s25]
            scf.yield %w26 : !air.async.token
          } else {
            %e26 = air.wait_all async [%s25]
            scf.yield %e26 : !air.async.token
          }
          %s27 = scf.if %go -> (!air.async.token) {
            %w27 = air.wait_all async [%s26]
            scf.yield %w27 : !air.async.token
          } else {
            %e27 = air.wait_all async [%s26]
            scf.yield %e27 : !air.async.token
          }
          %s28 = scf.if %go -> (!air.async.token) {
            %w28 = air.wait_all async [%s27]
            scf.yield %w28 : !air.async.token
          } else {
            %e28 = air.wait_all async [%s27]
            scf.yield %e28 : !air.async.token
          }
          %s29 = scf.if %go -> (!air.async.token) {
            %w29 = air.wait_all async [%s28]
            scf.yield %w29 : !air.async.token
          } else {
            %e29 = air.wait_all async [%s28]
            scf.yield %e29 : !air.async.token
          }
          %s30 = scf.if %go -> (!air.async.token) {
            %w30 = air.wait_all async [%s29]
            scf.yield %w30 : !air.async.token
          } else {
            %e30 = air.wait_all async [%s29]
            scf.yield %e30 : !air.async.token
          }
          %s31 = scf.if %go -> (!air.async.token) {
            %w31 = air.wait_all async [%s30]
            scf.yield %w31 : !air.async.token
          } else {
            %e31 = air.wait_all async [%s30]
            scf.yield %e31 : !air.async.token
          }
          %s32 = scf.if %go -> (!air.async.token) {
            %w32 = air.wait_all async [%s31]
            scf.yield %w32 : !air.async.token
          } else {
            %e32 = air.wait_all async [%s31]
            scf.yield %e32 : !air.async.token
          }
          %s33 = scf.if %go -> (!air.async.token) {
            %w33 = air.wait_all async [%s32]
            scf.yield %w33 : !air.async.token
          } else {
            %e33 = air.wait_all async [%s32]
            scf.yield %e33 : !air.async.token
          }
          %s34 = scf.if %go -> (!air.async.token) {
            %w34 = air.wait_all async [%s33]
            scf.yield %w34 : !air.async.token
          } else {
            %e34 = air.wait_all async [%s33]
            scf.yield %e34 : !air.async.token
          }
          %s35 = scf.if %go -> (!air.async.token) {
            %w35 = air.wait_all async [%s34]
            scf.yield %w35 : !air.async.token
          } else {
            %e35 = air.wait_all async [%s34]
            scf.yield %e35 : !air.async.token
          }
          %s36 = scf.if %go -> (!air.async.token) {
            %w36 = air.wait_all async [%s35]
            scf.yield %w36 : !air.async.token
          } else {
            %e36 = air.wait_all async [%s35]
            scf.yield %e36 : !air.async.token
          }
          %s37 = scf.if %go -> (!air.async.token) {
            %w37 = air.wait_all async [%s36]
            scf.yield %w37 : !air.async.token
          } else {
            %e37 = air.wait_all async [%s36]
            scf.yield %e37 : !air.async.token
          }
          %s38 = scf.if %go -> (!air.async.token) {
            %w38 = air.wait_all async [%s37]
            scf.yield %w38 : !air.async.token
          } else {
            %e38 = air.wait_all async [%s37]
            scf.yield %e38 : !air.async.token
          }
          %s39 = scf.if %go -> (!air.async.token) {
            %w39 = air.wait_all async [%s38]
            scf.yield %w39 : !air.async.token
          } else {
            %e39 = air.wait_all async [%s38]
            scf.yield %e39 : !air.async.token
          }
          %s40 = scf.if %go -> (!air.async.token) {
            %w40 = air.wait_all async [%s39]
            scf.yield %w40 : !air.async.token
          } else {
            %e40 = air.wait_all async [%s39]
            scf.yield %e40 : !air.async.token
          }
          scf.yield %s40 : !air.async.token
        }
        %g1 = air.channel.get async @channel_13[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}
