//===- channel_transfer_same_buffer.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -split-input-file | FileCheck %s

// A put and a get that move data to each other, through one buffer. Once both
// are async, a token edge between them deadlocks: the get waits for the put to
// finish, and the put for the get to take its data.

// Each element is read and written at the same stream step, so the transfer
// orders the two accesses itself: no edge between them.

// CHECK-LABEL: func.func @same_place
// CHECK: air.channel.put async @C[]
// CHECK-NEXT: air.channel.get async @C[]
module {
  air.channel @C []
  func.func @same_place(%x : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.channel.put @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    air.channel.get @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    return
  }
}

// -----

// The get writes %x[0..2) at stream steps 0-1, from %a; the second put reads
// %x[0..2) at steps 2-3. With both ends async no ordering is correct, so the
// get stays synchronous, which buffers its transfer, after a blocking wait for
// the put that reads %x.

// CHECK-LABEL: func.func @needs_buffer
// CHECK: air.channel.put async @C[]
// CHECK: %[[PX:.*]] = air.channel.put async [%{{.*}}] @C[]
// CHECK-NEXT: air.wait_all [%[[PX]]]
// CHECK-NEXT: air.channel.get @C[]
module {
  air.channel @C []
  func.func @needs_buffer(%a : memref<4xi32>, %x : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    air.channel.put @C[] (%a[%c0] [%c2] [%c1]) : (memref<4xi32>)
    air.channel.put @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    air.channel.get @C[] (%x[%c0] [%c4] [%c1]) : (memref<4xi32>)
    return
  }
}

// -----

// The first get writes %x[0..2) at steps 0-1 though the put before it reads
// %x[0..2) at steps 2-3, for the second get. Keeping only the first get
// synchronous would wait for a put that waits for the second get: both stay
// synchronous, and every transfer on the slot is buffered.

// CHECK-LABEL: func.func @two_gets_need_buffer
// CHECK: air.channel.put async @C[]
// CHECK: %[[PX:.*]] = air.channel.put async [%{{.*}}] @C[]
// CHECK-NEXT: air.wait_all [%[[PX]]]
// CHECK-NEXT: air.channel.get @C[]
// CHECK-NEXT: air.wait_all [%[[PX]]]
// CHECK-NEXT: air.channel.get @C[]
module {
  air.channel @C []
  func.func @two_gets_need_buffer(%a : memref<4xi32>, %x : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.channel.put @C[] (%a[%c0] [%c2] [%c1]) : (memref<4xi32>)
    air.channel.put @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    air.channel.get @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    air.channel.get @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    return
  }
}

// -----

// Endpoints in a loop: their stream steps are not tracked, and the memory
// tracing orders them as before.

// CHECK-LABEL: func.func @in_loop
// CHECK: %[[P:.*]] = air.channel.put async [%{{.*}}] @C[]
// CHECK: air.channel.get async [{{.*}}%[[P]]{{.*}}] @C[]
module {
  air.channel @C []
  func.func @in_loop(%x : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      air.channel.put @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
      air.channel.get @C[] (%x[%c0] [%c2] [%c1]) : (memref<4xi32>)
    }
    return
  }
}
