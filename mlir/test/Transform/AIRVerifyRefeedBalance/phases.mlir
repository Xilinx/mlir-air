//===- phases.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -verify-diagnostics

// The rates must be checked per cyclo-static phase, not per dispatch.
//
// Four dispatch iterations pick an arm from `wave < 2`, exactly as a
// superkernel does. Each arm produces 64 * refeed 2 = 128 tokens:
//
//   arm "case 0" (waves 2,3): consumes 1 x 64  ->  surplus 64 per wave
//   arm "default" (waves 0,1): consumes 3 x 64 ->  deficit 64 per wave
//
// Summed over the dispatch the two cancel exactly -- 512 produced, 512
// consumed -- so a plain SDF check sees a balanced channel and reports
// nothing. Only the per-arm equation exposes the starved arm.

air.channel @arms [1] {air.refeed_count = 2 : i32}

func.func @per_arm_rates() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  scf.for %wave = %c0 to %c4 step %c1 {
    %lt = arith.cmpi slt, %wave, %c2 : index
    %ext = arith.extui %lt : i1 to i32
    %arm = arith.index_cast %ext : i32 to index
    scf.index_switch %arm
    case 0 {
      // expected-warning @+3 {{air.channel @arms[0] is unbalanced}}
      // expected-note @+2 {{the balance closes at air.refeed_count = 1}}
      // expected-note @+1 {{producer: 64 tokens x refeed 2}}
      air.channel.put @arms[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
      // expected-note @+1 {{consumer: 64 tokens}}
      air.channel.get @arms[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
      scf.yield
    }
    default {
      // expected-error @+3 {{air.channel @arms[0] is unbalanced}}
      // expected-note @+2 {{the balance closes at air.refeed_count = 3}}
      // expected-note @+1 {{producer: 64 tokens x refeed 2}}
      air.channel.put @arms[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
      scf.for %i = %c0 to %c3 step %c1 {
        // expected-note @+1 {{consumer: 192 tokens}}
        air.channel.get @arms[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
      }
      scf.yield
    }
  }
  return
}
