//===- loop_fusion_waiter_between_loops.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-loop-fusion -verify-diagnostics | FileCheck %s

// An air.wait_all between two fusable loops waits on the first of them. The
// fused loop is created after the second loop, so the wait_all has to follow
// the fused loop, and so must the op that waits on the wait_all.

// CHECK-LABEL: func.func @waiter_between_loops
// CHECK: %[[FUSED:.*]] = scf.for
// CHECK: %[[WAIT:.*]] = air.wait_all async [{{.*}}%[[FUSED]]]

air.channel @chan_get [1, 1]
air.channel @chan_put [1, 1]

func.func @waiter_between_loops() {
  %c4 = arith.constant 4 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) {
    %1 = air.segment @seg async {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %t0, %buf0 = air.execute -> (memref<64x1024xbf16, 1>) {
        %a = memref.alloc() : memref<64x1024xbf16, 1>
        air.execute_terminator %a : memref<64x1024xbf16, 1>
      }
      %t1, %buf1 = air.execute -> (memref<64x1024xbf16, 1>) {
        %a = memref.alloc() : memref<64x1024xbf16, 1>
        air.execute_terminator %a : memref<64x1024xbf16, 1>
      }
      %start = air.wait_all async [%t0, %t1]
      %g0 = scf.for %i = %c0 to %c1024 step %c256 iter_args(%d = %start) -> (!air.async.token) {
        %e = air.channel.get async [%d] @chan_get[%c0, %c0] (%buf0[%c0, %i] [64, 256] [1024, 1]) : (memref<64x1024xbf16, 1>)
        scf.yield %e : !air.async.token
      }
      %w0 = air.wait_all async [%t0, %g0]
      %g1 = scf.for %i = %c0 to %c1024 step %c256 iter_args(%d = %start) -> (!air.async.token) {
        %e = air.channel.get async [%d] @chan_get[%c0, %c0] (%buf1[%c0, %i] [64, 256] [1024, 1]) : (memref<64x1024xbf16, 1>)
        scf.yield %e : !air.async.token
      }
      %p0 = scf.for %i = %c0 to %c1024 step %c256 iter_args(%d = %w0) -> (!air.async.token) {
        %e = air.channel.put async [%d] @chan_put[%c0, %c0] (%buf0[%c0, %i] [64, 256] [1024, 1]) : (memref<64x1024xbf16, 1>)
        scf.yield %e : !air.async.token
      }
      %p1 = scf.for %i = %c0 to %c1024 step %c256 iter_args(%d = %g1) -> (!air.async.token) {
        %e = air.channel.put async [%d] @chan_put[%c0, %c0] (%buf1[%c0, %i] [64, 256] [1024, 1]) : (memref<64x1024xbf16, 1>)
        scf.yield %e : !air.async.token
      }
      %t2 = air.execute [%p0, %p1] {
        memref.dealloc %buf0 : memref<64x1024xbf16, 1>
      }
      %t3 = air.execute [%p0, %p1] {
        memref.dealloc %buf1 : memref<64x1024xbf16, 1>
      }
    }
  }
  return
}
