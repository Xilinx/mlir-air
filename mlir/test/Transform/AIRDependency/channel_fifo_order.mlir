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

// Different indices => different FIFO sub-channels => left independent.

// CHECK-LABEL: func.func @chan_diff_index
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}]  @channel_1[%c0
// CHECK: air.channel.get async [%{{[a-z_0-9]+}}]  @channel_1[%c1

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
