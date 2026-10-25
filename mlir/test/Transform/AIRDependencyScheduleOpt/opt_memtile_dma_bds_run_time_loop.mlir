//===- opt_memtile_dma_bds_run_time_loop.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" | FileCheck %s

// A memtile loop whose trip count is known only at run time cannot become a
// dim of the put: a memtile descriptor has no size of run-time value. The
// loop stays.

// CHECK-LABEL: func.func @run_time_trips
// CHECK: scf.for {{.*}} to %{{.*}} step
// CHECK: air.channel.put {{.*}} @chan[
module {
  air.channel @chan [1, 1]
  func.func @run_time_trips(%n: index) {
    %0 = air.launch async () in () args(%ln=%n) : index {
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %t0, %buf = air.execute -> (memref<4096xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4096xbf16, 1 : i32>
          air.execute_terminator %a : memref<4096xbf16, 1 : i32>
        }
        %2 = scf.for %k = %c0 to %sn step %c1 iter_args(%d = %t0) -> (!air.async.token) {
          %o = arith.muli %k, %c64 : index
          %p = air.channel.put async [%d] @chan[%c0, %c0] (%buf[%o] [%c64] [%c1]) : (memref<4096xbf16, 1 : i32>)
          scf.yield %p : !air.async.token
        }
      }
    }
    return
  }
}
