//===- convergent.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -split-input-file -verify-diagnostics

// A convergent channel: several producers re-feed one consumer loop, each with
// its own refeed count. This is the shape the Q4NX decode superkernels use for
// their X feed (@xnorm), and it bounds how much the compiler can derive.
//
// The balance equation is one equation per edge, so it pins down the WEIGHTED
// SUM of the producers' counts, not the individual counts. Reduced from the
// llama-3.2-1b decode, where the real numbers are
//
//   2048x4 + 8192x4 + 2048x6 + 2048x32 = 118784 = 512 x 232 consumed
//
// and the split into (16, 64, 24, 128) consumer rounds -- which is what fixes
// each count -- is the design's per-phase round table. That table lives in the
// generator, not in the IR: the consumer is one flat loop, so nothing here
// distinguishes which rounds read which producer.
//
// So the pass stays SILENT on a convergent edge whose total closes, and does
// not volunteer a per-producer count. An inference that guessed one would be
// wrong here in a way no diagnostic could catch, which is what this guards.
air.channel @convergent [1] {air.refeed_count = 6 : i32}
func.func @several_producers_one_consumer_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c232 = arith.constant 232 : index
  %big = memref.alloc() : memref<8192xbf16, 1 : i32>
  %small = memref.alloc() : memref<2048xbf16, 1 : i32>
  %dst = memref.alloc() : memref<512xbf16, 2 : i32>

  // 2048 x 4 = 8192, via an alloc carrier.
  %tok, %staged = air.execute -> (memref<2048xbf16, 1 : i32>) {
    %m = memref.alloc() {air.refeed_count = 4 : i32} : memref<2048xbf16, 1 : i32>
    air.execute_terminator %m : memref<2048xbf16, 1 : i32>
  }
  air.channel.put @convergent[] (%staged[0] [2048] [1]) : (memref<2048xbf16, 1 : i32>)
  // 8192 x 4 = 32768, per-emission override.
  air.channel.put @convergent[] (%big[0] [8192] [1]) {air.refeed_count = 4 : i32} : (memref<8192xbf16, 1 : i32>)
  // 2048 x 6 = 12288, inheriting the channel default.
  air.channel.put @convergent[] (%small[0] [2048] [1]) : (memref<2048xbf16, 1 : i32>)
  // 2048 x 32 = 65536, per-emission override.
  air.channel.put @convergent[] (%small[0] [2048] [1]) {air.refeed_count = 32 : i32} : (memref<2048xbf16, 1 : i32>)

  scf.for %i = %c0 to %c232 step %c1 {
    air.channel.get @convergent[] (%dst[0] [512] [1]) : (memref<512xbf16, 2 : i32>)
  }
  return
}

// -----

// The sum is what is checked, so a convergent edge is still caught when the
// total is wrong -- here the last producer's count is 31 rather than 32, and
// 2048 tokens go missing. The note reports the whole-edge ratio; it cannot say
// which of the producers is the one that drifted.
air.channel @convergentShort [1] {air.refeed_count = 6 : i32}
func.func @convergent_deficit_is_still_caught() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c232 = arith.constant 232 : index
  %big = memref.alloc() : memref<8192xbf16, 1 : i32>
  %small = memref.alloc() : memref<2048xbf16, 1 : i32>
  %dst = memref.alloc() : memref<512xbf16, 2 : i32>

  // expected-error @+3 {{air.channel @convergentShort[0] is unbalanced}}
  // expected-note @+2 {{no integer air.refeed_count closes the balance}}
  // expected-note @+1 {{producer: 8192 tokens x refeed 4}}
  air.channel.put @convergentShort[] (%big[0] [8192] [1]) {air.refeed_count = 4 : i32} : (memref<8192xbf16, 1 : i32>)
  // expected-note @+1 {{producer: 2048 tokens x refeed 6}}
  air.channel.put @convergentShort[] (%small[0] [2048] [1]) : (memref<2048xbf16, 1 : i32>)
  // expected-note @+1 {{producer: 2048 tokens x refeed 31}}
  air.channel.put @convergentShort[] (%small[0] [2048] [1]) {air.refeed_count = 31 : i32} : (memref<2048xbf16, 1 : i32>)

  scf.for %i = %c0 to %c232 step %c1 {
    // expected-note @+1 {{consumer: 118784 tokens}}
    air.channel.get @convergentShort[] (%dst[0] [512] [1]) : (memref<512xbf16, 2 : i32>)
  }
  return
}
