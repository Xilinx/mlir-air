//===- unbalanced.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -split-input-file -verify-diagnostics

// Channels whose token rates do not close.

// A deficit starves the consumers and deadlocks the array: the refeed count
// says 2 re-sends but the consumer reads 4 times. The note names the count
// that would close it.
air.channel @short [1] {air.refeed_count = 2 : i32}
func.func @deficit() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  // expected-error @+3 {{air.channel @short[0] is unbalanced}}
  // expected-note @+2 {{the balance closes at air.refeed_count = 4}}
  // expected-note @+1 {{producer: 64 tokens x refeed 2}}
  air.channel.put @short[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c4 step %c1 {
    // expected-note @+1 {{consumer: 256 tokens}}
    air.channel.get @short[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}

// -----

// A surplus only wastes bandwidth, so it is a warning rather than an error.
air.channel @over [1] {air.refeed_count = 8 : i32}
func.func @surplus() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  // expected-warning @+3 {{air.channel @over[0] is unbalanced}}
  // expected-note @+2 {{the balance closes at air.refeed_count = 2}}
  // expected-note @+1 {{producer: 64 tokens x refeed 8}}
  air.channel.put @over[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c2 step %c1 {
    // expected-note @+1 {{consumer: 128 tokens}}
    air.channel.get @over[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}

// -----

// No integer refeed count divides the consumed tokens by the produced ones;
// the pass says so instead of rounding to something plausible.
air.channel @ragged [1]
func.func @not_an_integer_ratio() {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<48xbf16, 2 : i32>
  // expected-error @+3 {{air.channel @ragged[0] is unbalanced}}
  // expected-note @+2 {{no integer air.refeed_count closes the balance}}
  // expected-note @+1 {{producer: 64 tokens x refeed 1}}
  air.channel.put @ragged[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c3 step %c1 {
    // expected-note @+1 {{consumer: 144 tokens}}
    air.channel.get @ragged[] (%dst[] [] []) : (memref<48xbf16, 2 : i32>)
  }
  return
}

// -----

// Two edges of one bundle, unbalanced by the same amount. Reports are collapsed
// when a mode repeats across dispatch iterations, but distinct edges are
// distinct findings: keying that de-duplication on the numbers alone would
// silently drop the second one.
air.channel @twin [2] {air.refeed_count = 4 : i32}
func.func @identical_imbalance_on_two_edges() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  // expected-warning @+3 {{air.channel @twin[0] is unbalanced}}
  // expected-note @+2 {{the balance closes at air.refeed_count = 2}}
  // expected-note @+1 {{producer: 64 tokens x refeed 4}}
  air.channel.put @twin[%c0] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  // expected-warning @+3 {{air.channel @twin[1] is unbalanced}}
  // expected-note @+2 {{the balance closes at air.refeed_count = 2}}
  // expected-note @+1 {{producer: 64 tokens x refeed 4}}
  air.channel.put @twin[%c1] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c2 step %c1 {
    // expected-note @+1 {{consumer: 128 tokens}}
    air.channel.get @twin[%c0] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
    // expected-note @+1 {{consumer: 128 tokens}}
    air.channel.get @twin[%c1] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}
