//===- hoisted_loop_raw_dependency.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-dependency -air-dma-to-channel %s | FileCheck %s

// Two external (L3) channel ops in a segment: a loop of gets that fills a
// buffer, and a put that reads a region of the same buffer afterwards.
// air-dependency makes the put depend on the loop. air-dma-to-channel hoists
// the external ops to the launch one at a time, so when the put is hoisted the
// gets already live in a hoisted loop. The put must depend on that loop's
// token, or nothing orders the read after the writes that fill the buffer.

// The put depends on the hoisted loop that holds the gets.
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK:   air.channel.get async {{.*}}@Drain
// CHECK: air.channel.put async [{{.*}}%[[LOOP]]{{.*}}] @FaIn

module {
  air.channel @Drain [1]
  air.channel @FaIn [1]
  func.func @f(%qkv: memref<256x1600xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%x=%qkv) : memref<256x1600xbf16> {
      air.segment @s args(%y=%x) : memref<256x1600xbf16> {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1s = arith.constant 1 : index
        scf.for %i = %c0 to %c3 step %c1s {
          %r = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%i]
          air.channel.get @Drain[%c0] (%y[%r, 0] [32, 320] [1600, 1]) : (memref<256x1600xbf16>)
        }
        air.channel.put @FaIn[%c0] (%y[0, 0] [64, 64] [1600, 1]) : (memref<256x1600xbf16>)
      }
    }
    return
  }
}
