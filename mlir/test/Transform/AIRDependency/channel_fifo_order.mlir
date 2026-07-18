//===- channel_fifo_order.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-enforce-channel-fifo-order | FileCheck %s

// A single air.channel is an ordered FIFO. Two async gets on the same channel +
// same index + same direction that touch DIFFERENT buffers (e.g. two phases of
// a kernel temporally reusing one channel, after fusion collapsed the loops)
// are left unordered by the dependency analysis. This pass adds a direct async
// dependency from the second to the first so they are serialized. It only orders
// same-block ops.

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
