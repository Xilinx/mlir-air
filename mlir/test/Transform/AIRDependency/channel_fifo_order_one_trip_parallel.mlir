//===- channel_fifo_order_one_trip_parallel.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Puts on one channel, each in a parallel loop of one iteration (as hoisting a
// core's transfers out of its herd leaves them): such a loop runs its body
// once, so the second put waits for the first loop, as it would for the put.

// CHECK-LABEL: func.func @one_trip_nests
// CHECK: %[[P0:.*]] = scf.parallel
// CHECK: air.channel.put async {{.*}}@c
// CHECK: scf.parallel
// CHECK: air.channel.put async [{{.*}}%[[P0]]{{.*}}] @c
module {
  air.channel @c [1]
  func.func @one_trip_nests(%arg0: memref<8x256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<8x256xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      scf.parallel (%i, %j) = (%c0, %c0) to (%c1_0, %c1_0) step (%c1_0, %c1_0) {
        air.channel.put @c[] (%a[%c0, %c0] [%c1_0, %c1_0] [%c1_0, %c1_0]) : (memref<8x256xi32>)
        scf.reduce
      }
      scf.parallel (%i, %j) = (%c0, %c0) to (%c1_0, %c1_0) step (%c1_0, %c1_0) {
        air.channel.put @c[] (%a[%c1_0, %c0] [%c1_0, %c1_0] [%c1_0, %c1_0]) : (memref<8x256xi32>)
        scf.reduce
      }
    }
    return
  }
}
