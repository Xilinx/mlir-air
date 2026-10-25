//===- specialize-channel-wrap-and-stride-dest.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride=scope=func -split-input-file | FileCheck %s

// Puts with a packet destination.

// Dropping the unit dim keeps the destination.

// CHECK-LABEL: func.func @unit_dim
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c0{{.*}}] [%c8192{{.*}}] [%c1{{.*}}]) dest(%c1{{.*}})

module {
  air.channel @ab [1, 1]
  func.func @unit_dim(%src: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %k = air.channel.put async [%0] @ab[] (%src[%c0, %c0] [%c1, %c8192] [%c8192, %c1]) dest(%c1) : (memref<*xbf16>)
    return
  }
}

// -----

// The destination is computed in the loop, the same on every trip: the loop
// folds, and the destination is computed before the folded put.

// CHECK-LABEL: func.func @invariant_dest
// CHECK: %[[D:.*]] = arith.addi %arg1, %c1{{.*}} : index
// CHECK-NEXT: air.channel.put {{.*}} @ab[] (%arg0[%c0{{.*}}] [%c32768{{.*}}] [%c1{{.*}}]) dest(%[[D]])
// CHECK-NOT: scf.for

module {
  air.channel @ab [1, 1]
  func.func @invariant_dest(%src: memref<*xbf16>, %j: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c0 to %c4 step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c8192 : index
      %d = arith.addi %j, %c1 : index
      %k = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) dest(%d) : (memref<*xbf16>)
      scf.yield %k : !air.async.token
    }
    return
  }
}

// -----

// A destination that changes with the trip is no dim of the transfer: the
// loop does not fold, and unrolls into puts that each keep their own.

// CHECK-LABEL: func.func @varying_dest
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c0{{.*}}] [%c8192{{.*}}] [%c1{{.*}}]) dest(%c0
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c8192{{.*}}] [%c8192{{.*}}] [%c1{{.*}}]) dest(%c1
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c16384{{.*}}] [%c8192{{.*}}] [%c1{{.*}}]) dest(%c2
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c24576{{.*}}] [%c8192{{.*}}] [%c1{{.*}}]) dest(%c3
// CHECK-NOT: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @varying_dest(%src: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c0 to %c4 step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c8192 : index
      %d = arith.addi %b, %c0 : index
      %k = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) dest(%d) : (memref<*xbf16>)
      scf.yield %k : !air.async.token
    }
    return
  }
}
