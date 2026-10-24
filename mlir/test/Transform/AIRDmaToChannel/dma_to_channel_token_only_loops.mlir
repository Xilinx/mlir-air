//===- dma_to_channel_token_only_loops.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// Loops in a segment that only pass their own tokens along are replaced by a
// wait on their init tokens. A loop that also waits for a token from outside
// it, or that holds a blocking wait, is kept: replacing it would drop that
// dependency or that synchronization.

// CHECK-LABEL: func.func @token_only_loops
// CHECK: air.segment
// CHECK: %[[INIT:.*]] = air.wait_all async
// CHECK: air.execute
// CHECK: %[[PLAIN:.*]] = air.wait_all async [%[[INIT]]]
// CHECK-NEXT: %[[OUTSIDE:.*]] = scf.for {{.*}} iter_args(%[[D:.*]] = %[[PLAIN]])
// CHECK-NEXT: air.wait_all async [%[[D]], %{{.*}}]
// CHECK: scf.for {{.*}} iter_args(%[[D2:.*]] = %[[OUTSIDE]])
// CHECK-NEXT: air.wait_all [%[[D2]]]

func.func @token_only_loops(%arg0: memref<16xi32>) {
  %c1 = arith.constant 1 : index
  air.launch () in () {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %t0 = air.wait_all async
      %te, %b = air.execute -> (memref<16xi32, 1>) {
        %a = memref.alloc() : memref<16xi32, 1>
        air.execute_terminator %a : memref<16xi32, 1>
      }
      %plain = scf.for %i = %c0 to %c4 step %c1_s iter_args(%d = %t0) -> (!air.async.token) {
        %w = air.wait_all async [%d]
        scf.yield %w : !air.async.token
      }
      %outside = scf.for %i = %c0 to %c4 step %c1_s iter_args(%d = %plain) -> (!air.async.token) {
        %w = air.wait_all async [%d, %te]
        scf.yield %w : !air.async.token
      }
      %blocking = scf.for %i = %c0 to %c4 step %c1_s iter_args(%d = %outside) -> (!air.async.token) {
        air.wait_all [%d]
        %w = air.wait_all async [%d]
        scf.yield %w : !air.async.token
      }
      %dd = air.execute [%blocking] {
        memref.dealloc %b : memref<16xi32, 1>
      }
    }
  }
  return
}
