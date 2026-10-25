//===- opt_shim_dma_bds_run_time_loops_kept.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// Loops of run-time trip count or around run-time sizes that the shim pass
// leaves as they are.

// A memtile descriptor has no run-time length: a dense run of an L2 buffer
// over a run-time number of trips, inside the segment, stays a loop.

// CHECK-LABEL: func.func @l2_run_time_trips
// CHECK: air.segment
// CHECK-NOT: scf.if
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %{{.*}} step %c1
// CHECK-NEXT: arith.muli
// CHECK-NEXT: air.channel.put {{.*}} @l2[] (%{{.*}}[%{{.*}}] [%c256{{.*}}] [%c1{{.*}}])

module {
  air.channel @l2 [1, 1]
  func.func @l2_run_time_trips(%n: index) {
    %c1 = arith.constant 1 : index
    %t = air.launch async (%x) in (%sx = %c1) args(%ln = %n) : index {
      %s = air.segment @seg async args(%m = %ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        %ta, %buf = air.execute -> (memref<65536xbf16, 1>) {
          %b = memref.alloc() : memref<65536xbf16, 1>
          air.execute_terminator %b : memref<65536xbf16, 1>
        }
        %1 = scf.for %b = %c0_s to %m step %c1_s iter_args(%u = %ta) -> (!air.async.token) {
          %o = arith.muli %b, %c256 : index
          %k = air.channel.put async [%u] @l2[] (%buf[%o] [%c256] [%c1_s]) : (memref<65536xbf16, 1>)
          scf.yield %k : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Nor does a static loop around a put of an L2 buffer of run-time size become
// its repeat.

// CHECK-LABEL: func.func @l2_run_time_size
// CHECK: air.segment
// CHECK-NOT: [%c4{{.*}}, %{{.*}}] [%c0{{.*}}, %c1{{.*}}]
// CHECK: air.channel.put {{.*}} @l2[] (%{{.*}}[%c0{{.*}}] [%{{.*}}] [%c1{{.*}}])

module {
  air.channel @l2 [1, 1]
  func.func @l2_run_time_size(%n: index) {
    %c1 = arith.constant 1 : index
    %t = air.launch async (%x) in (%sx = %c1) args(%ln = %n) : index {
      %s = air.segment @seg async args(%m = %ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %ta, %buf = air.execute -> (memref<65536xbf16, 1>) {
          %b = memref.alloc() : memref<65536xbf16, 1>
          air.execute_terminator %b : memref<65536xbf16, 1>
        }
        %1 = scf.for %b = %c0_s to %c4_s step %c1_s iter_args(%u = %ta) -> (!air.async.token) {
          %k = air.channel.put async [%u] @l2[] (%buf[%c0_s] [%m] [%c1_s]) : (memref<65536xbf16, 1>)
          scf.yield %k : !air.async.token
        }
      }
    }
    return
  }
}
