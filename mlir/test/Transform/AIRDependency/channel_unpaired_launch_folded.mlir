//===- channel_unpaired_launch_folded.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dependency-canonicalize 2>&1 | FileCheck %s

// The segment receives from the launch in a loop that runs zero times for
// this function's sizes; the launch's matching loop folded away with its put.
// The segment body is shared with functions that do put to the channel, so
// the get alone is not an error.
// CHECK-NOT: not in pairs
// CHECK: func.func @folded_launch_put
// CHECK: air.channel.get
air.channel @chan [1, 1]
func.func @folded_launch_put(%n: index) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a0, %a1) in (%a2=%c1, %a3=%c1) args(%an=%n) : index {
    %1 = air.segment async args(%sn=%an) : index {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %t0, %buf = air.execute -> (memref<32x32xbf16, 1>) {
        %a = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %a : memref<32x32xbf16, 1>
      }
      scf.for %i = %c0 to %sn step %c1_s {
        %t1 = air.channel.get async @chan[] (%buf[] [] []) : (memref<32x32xbf16, 1>)
      }
      %td = air.execute {
        memref.dealloc %buf : memref<32x32xbf16, 1>
      }
    }
  }
  return
}
