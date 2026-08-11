//===- balanced.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -split-input-file -verify-diagnostics

// Channels whose token rates close. The pass must stay silent.

// A producer re-broadcasting its buffer 4 times feeds four consumer reads.
air.channel @refeed [1] {air.refeed_count = 4 : i32}
func.func @closes_with_refeed() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  air.channel.put @refeed[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.get @refeed[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}

// -----

// No refeed: one put, one get, same size.
air.channel @plain [1]
func.func @closes_without_refeed() {
  %src = memref.alloc() : memref<32xbf16, 1 : i32>
  %dst = memref.alloc() : memref<32xbf16, 2 : i32>
  air.channel.put @plain[] (%src[] [] []) : (memref<32xbf16, 1 : i32>)
  air.channel.get @plain[] (%dst[] [] []) : (memref<32xbf16, 2 : i32>)
  return
}

// -----

// Each edge of a bundle is its own equation; both close.
air.channel @bundle [2]
func.func @bundle_edges_close() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<16xbf16, 1 : i32>
  %dst = memref.alloc() : memref<16xbf16, 2 : i32>
  air.channel.put @bundle[%c0] (%src[] [] []) : (memref<16xbf16, 1 : i32>)
  air.channel.put @bundle[%c1] (%src[] [] []) : (memref<16xbf16, 1 : i32>)
  air.channel.get @bundle[%c0] (%dst[] [] []) : (memref<16xbf16, 2 : i32>)
  air.channel.get @bundle[%c1] (%dst[] [] []) : (memref<16xbf16, 2 : i32>)
  return
}
