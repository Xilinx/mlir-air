//===- carriers.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -split-input-file -verify-diagnostics

// The refeed count reaches a put by three different routes, and the analysis
// must resolve the same one the lowering will. Getting the precedence wrong
// silently changes every rate on the channel.

// Route 1: the channel declaration, the default for every emission on it.
air.channel @viaChannel [1] {air.refeed_count = 3 : i32}
func.func @channel_declaration() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  air.channel.put @viaChannel[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c3 step %c1 {
    air.channel.get @viaChannel[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}

// -----

// Route 2: a per-emission override beats the declaration. A value of 1 is the
// interesting case -- it exists precisely to cancel an inherited default, so
// it must not be mistaken for "no attribute, fall through to the channel".
air.channel @cancelled [1] {air.refeed_count = 8 : i32}
func.func @per_put_override_of_one_cancels() {
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  air.channel.put @cancelled[] (%src[] [] []) {air.refeed_count = 1 : i32} : (memref<64xbf16, 1 : i32>)
  air.channel.get @cancelled[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  return
}

// -----

// Route 3: the L2 rendezvous buffer the put reads from. The attribute sits on
// the memref.alloc inside an air.execute -- which is what air-to-aie copies
// onto the aie.buffer -- so it has to be found through the execute, not on the
// execute itself.
air.channel @viaAlloc [1]
func.func @alloc_carrier_through_execute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  %tok, %buf = air.execute -> (memref<64xbf16, 1 : i32>) {
    %m = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1 : i32>
    air.execute_terminator %m : memref<64xbf16, 1 : i32>
  }
  air.channel.put @viaAlloc[] (%buf[] [] []) : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.get @viaAlloc[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}

// -----

// A per-emission override also beats the alloc carrier.
air.channel @bothCarriers [1]
func.func @put_attr_beats_alloc_carrier() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  %tok, %buf = air.execute -> (memref<64xbf16, 1 : i32>) {
    %m = memref.alloc() {air.refeed_count = 8 : i32} : memref<64xbf16, 1 : i32>
    air.execute_terminator %m : memref<64xbf16, 1 : i32>
  }
  air.channel.put @bothCarriers[] (%buf[] [] []) {air.refeed_count = 2 : i32} : (memref<64xbf16, 1 : i32>)
  scf.for %i = %c0 to %c2 step %c1 {
    air.channel.get @bothCarriers[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  }
  return
}
