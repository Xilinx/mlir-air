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

// A loop is a scope boundary: gets in distinct scf.for bodies stay unordered.

// CHECK-LABEL: func.func @chan_fifo_distinct_loops
// CHECK: scf.for
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_7
// CHECK: scf.for
// CHECK-NOT: air.channel.get async [%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}] @channel_7
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}] @channel_7

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
