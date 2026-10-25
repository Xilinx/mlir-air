//===- specialize-run-time-trip-count.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride=scope=func -split-input-file | FileCheck %s

// A loop of run-time trip count over a dense run folds into a dim of that
// size only where the count is known to be positive: a transfer of length
// zero or less is no transfer of nothing. With no such guard the loop stays.

// CHECK-LABEL: func.func @unguarded
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %arg1 step %c1
// CHECK: air.channel.put {{.*}} [%c8192{{.*}}] [%c1{{.*}}])

module {
  air.channel @ab [1, 1]
  func.func @unguarded(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c8192 : index
      %k = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) : (memref<*xbf16>)
      scf.yield %k : !air.async.token
    }
    return
  }
}

// -----

// Under a guard that the upper bound exceeds the lower one, it folds.

// CHECK-LABEL: func.func @guarded
// CHECK: scf.if
// CHECK-NEXT: air.channel.put {{.*}} @ab[] (%arg0[{{.*}}] [%arg1, %c8192{{.*}}] [%c8192{{.*}}, %c1{{.*}}])
// CHECK-NOT: scf.for

module {
  air.channel @ab [1, 1]
  func.func @guarded(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %any = arith.cmpi sgt, %n, %c0 : index
    %r = scf.if %any -> (!air.async.token) {
      %1 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
        %o = arith.muli %b, %c8192 : index
        %k = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) : (memref<*xbf16>)
        scf.yield %k : !air.async.token
      }
      scf.yield %1 : !air.async.token
    } else {
      scf.yield %0 : !air.async.token
    }
    return
  }
}
