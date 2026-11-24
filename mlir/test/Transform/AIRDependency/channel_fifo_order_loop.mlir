//===- channel_fifo_order_loop.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Puts on one channel before a loop and in it, reading one buffer: no memory
// dependence orders them, so the channel's FIFO order has to. The put in the
// loop waits for the put before it on its first iteration, and for its own
// previous iteration through the iteration argument.

// CHECK-LABEL: func.func @put_before_loop
// CHECK: %[[P0:.*]] = air.channel.put async {{.*}}@channel_2
// CHECK: scf.for {{.*}} iter_args(%[[A:.*]] = %{{[a-z_0-9]+}})
// CHECK: air.channel.put async [%[[P0]], %[[A]]] @channel_2
module {
  air.channel @channel_2 [1, 1]
  func.func @put_before_loop(%arg0 : memref<1024xi32>) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    scf.for %arg3 = %c0 to %c8 step %c1 {
      air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    }
    return
  }
}
