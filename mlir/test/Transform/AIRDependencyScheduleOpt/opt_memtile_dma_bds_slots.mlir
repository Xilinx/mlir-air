//===- opt_memtile_dma_bds_slots.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" | FileCheck %s

// A loop fills another slot of a memtile buffer each trip, and the buffer is
// drained slot by slot: one BD per slot (the loop unrolled), so a drained slot
// can refill while the others still drain, rather than one BD for the whole
// buffer.

// CHECK-LABEL: func.func @slots
// CHECK: air.channel.get {{.*}} @in[] (%{{.*}}[%c0{{.*}}] [%c128{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.get {{.*}} @in[] (%{{.*}}[%c128{{.*}}] [%c128{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.get {{.*}} @in[] (%{{.*}}[%c256{{.*}}] [%c128{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.get {{.*}} @in[] (%{{.*}}[%c384{{.*}}] [%c128{{.*}}] [%c1{{.*}}])

module {
  air.channel @in [1, 1]
  air.channel @out [1, 1]
  func.func @slots() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t, %buf = air.execute -> (memref<4x2x64xi16, 1>) {
          %a = memref.alloc() : memref<4x2x64xi16, 1>
          air.execute_terminator %a : memref<4x2x64xi16, 1>
        }
        %2 = scf.for %j = %c0 to %c4 step %c1_0 iter_args(%d = %t) -> (!air.async.token) {
          %g = air.channel.get async [%d] @in[] (%buf[%j, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
          scf.yield %g : !air.async.token
        }
        %3 = scf.for %j = %c0 to %c4 step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p = air.channel.put async [%d] @out[] (%buf[%j, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
          scf.yield %p : !air.async.token
        }
        %4 = air.execute [%3] {
          memref.dealloc %buf : memref<4x2x64xi16, 1>
        }
      }
    }
    return
  }
}
