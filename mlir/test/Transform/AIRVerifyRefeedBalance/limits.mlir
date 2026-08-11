//===- limits.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -split-input-file -verify-diagnostics

// What the analysis declines to judge, and how it declines.

// A rate it cannot resolve is skipped, not guessed at. The trip count here is
// a function argument, so no number is reported for this channel at all.
air.channel @dynamic [1] {air.refeed_count = 2 : i32}
func.func @unresolvable_trip_is_skipped(%n : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  air.channel.put @dynamic[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %n step %c1 {
    air.channel.get @dynamic[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}

// -----

// When one side of a channel is gated by the arm switch and the other is
// arm-independent common code, the per-phase equation is not well posed: the
// consumer loop runs in both arms while the producers differ. Such an edge is
// summed over the whole dispatch instead, and that weaker check never
// escalates to an error even though the totals show a shortfall.
air.channel @crossArm [1] {air.refeed_count = 2 : i32}
func.func @cross_arm_never_errors() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  scf.for %wave = %c0 to %c2 step %c1 {
    %lt = arith.cmpi slt, %wave, %c1 : index
    %ext = arith.extui %lt : i1 to i32
    %arm = arith.index_cast %ext : i32 to index
    scf.index_switch %arm
    case 0 {
      // expected-note @+1 {{producer: 64 tokens x refeed 2}}
      air.channel.put @crossArm[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
      scf.yield
    }
    default {
      // A deficit, yet reported as a warning: the whole-dispatch sum is a
      // weaker statement than a per-phase one and must not gate the build.
      // expected-warning @+3 {{unbalanced in the whole dispatch (its two sides are gated by different arms)}}
      // expected-note @+2 {{no integer air.refeed_count closes the balance}}
      // expected-note @+1 {{producer: 64 tokens x refeed 2}}
      air.channel.put @crossArm[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
      // expected-note @+1 {{producer: 64 tokens x refeed 2}}
      air.channel.put @crossArm[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
      scf.yield
    }
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-note @+1 {{consumer: 256 tokens}}
      air.channel.get @crossArm[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
    }
  }
  return
}
