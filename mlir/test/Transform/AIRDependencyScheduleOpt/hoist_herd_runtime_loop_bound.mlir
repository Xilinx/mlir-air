//===- hoist_herd_runtime_loop_bound.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests | FileCheck %s

// A herd in a loop whose bound is computed from a run-time value (`n * 2`)
// is hoisted out of the loop: the run-time value becomes a herd operand and
// the bound is recomputed inside the herd, so the herd's loop keeps its form.
// CHECK-LABEL: func.func @runtime_bound
// CHECK: air.segment @seg async args(%[[SN:.*]]=%{{.*}}) : index
// CHECK: air.herd @herd_0 async {{.*}} args(%[[HN:.*]]=%[[SN]]) : index
// CHECK: %[[UB:.*]] = arith.muli %[[HN]], %{{.*}} : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[UB]] step
func.func @runtime_bound(%n: index) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%l) in (%ls=%c1) args(%ln=%n) : index {
    %s = air.segment @seg async args(%sn=%ln) : index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %ub = arith.muli %sn, %c2 : index
      %w = air.wait_all async
      %r = scf.for %i = %c0 to %ub step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %h = air.herd @herd_0 async [%d] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %t0, %b = air.execute -> (memref<16xi32, 2>) {
            %a = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %a : memref<16xi32, 2>
          }
          %t1 = air.execute [%t0] {
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
        scf.yield %h : !air.async.token
      }
    }
  }
  return
}
