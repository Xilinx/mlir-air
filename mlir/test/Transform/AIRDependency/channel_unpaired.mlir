//===- channel_unpaired.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that air-dependency-canonicalize does not crash when a channel op
// has no matching put/get counterpart (e.g. channel.get without channel.put).
// Previously this caused an assertion failure accessing an empty vector.

// RUN: air-opt %s -air-dependency -air-dependency-canonicalize 2>&1 | FileCheck %s
// CHECK: error: 'air.channel.get' op found channel op not in pairs

air.channel @chan [1, 1]
func.func @same_channel_different_depths() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a0, %a1) in (%a2=%c1, %a3=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @h async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<32x32xbf16, 2>) {
          %a = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %a : memref<32x32xbf16, 2>
        }
        %t1 = air.channel.get async [%t0] @chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        scf.for %i = %c0 to %c2 step %c1_h {
          %t2 = air.channel.get async @chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        }
        %td = air.execute {
          memref.dealloc %buf : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
