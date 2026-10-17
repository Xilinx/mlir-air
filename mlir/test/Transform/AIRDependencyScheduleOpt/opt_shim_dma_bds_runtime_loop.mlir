//===- opt_shim_dma_bds_runtime_loop.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -verify-diagnostics | FileCheck %s

// A launch-level loop whose trip count is known only at run time cannot be
// unrolled into a BD chain, so it is kept as a loop, without an error.

// CHECK-LABEL: func.func @runtime_trips
// CHECK: scf.for {{.*}} to %{{.*}} step
// CHECK: air.channel.put {{.*}} @in[]
// CHECK: air.channel.put {{.*}} @in[]
module {
  air.channel @in [1, 1]
  func.func @runtime_trips(%arg0: memref<4096xi16>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0, %ln=%n) : memref<4096xi16>, index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %t = air.wait_all async
      %2 = scf.for %k = %c0 to %ln step %c1_0 iter_args(%d = %t) -> (!air.async.token) {
        %o = arith.muli %k, %c64 : index
        %p0 = air.channel.put async [%d] @in[] (%a[%o] [%c64] [%c1_0]) : (memref<4096xi16>)
        %p1 = air.channel.put async [%p0, %d] @in[] (%a[%o] [%c64] [%c1_0]) : (memref<4096xi16>)
        scf.yield %p1 : !air.async.token
      }
      %1 = air.segment @seg async {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %t2, %buf = air.execute -> (memref<64xi16, 1>) {
          %b = memref.alloc() : memref<64xi16, 1>
          air.execute_terminator %b : memref<64xi16, 1>
        }
        %g = air.channel.get async [%t2] @in[] (%buf[] [] []) : (memref<64xi16, 1>)
      }
    }
    return
  }
}
