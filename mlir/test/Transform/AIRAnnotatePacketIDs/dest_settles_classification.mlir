//===- dest_settles_classification.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="report=true" -split-input-file -verify-diagnostics

// A put naming a `dest` settles space-vs-time by itself, so classification must
// not fall back to the volume heuristic -- which needs every put's trip count
// to be static and therefore cannot see a design that varies a count per arm.

// THE CASE THAT REGRESSED. The put's trip count is a runtime value, so no
// static volume exists for it; the channel is still a demux, because the put
// chooses its leaf at run time and nothing else it could mean.
module {
  // expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
  air.channel @dyn [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f(%n : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<512xbf16, 2 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 1 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 1 : i32>
    // Runtime trip count -- the volume discriminator cannot price this loop.
    scf.for %i = %c0 to %n step %c1 {
      air.channel.put @dyn[%c0, %c0] (%src[] [] []) dest(%c0) : (memref<512xbf16, 2 : i32>)
    }
    air.channel.put @dyn[%c0, %c0] (%src[] [] []) dest(%c1) : (memref<512xbf16, 2 : i32>)
    air.channel.get @dyn[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 1 : i32>)
    air.channel.get @dyn[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 1 : i32>)
    return
  }
}

// -----

// The heuristic still applies where it is the only evidence: no put names a
// dest here, the destinations replicate the stream, so this is a broadcast and
// one id serves both. Guards against the dest check swallowing this case.
module {
  // expected-remark @below {{broadcast over 2 destination(s); infers 1 routing id(s)}}
  air.channel @bcast [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @g() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<512xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @bcast[%c0, %c0] (%src[] [] []) : (memref<512xbf16, 1 : i32>)
    air.channel.get @bcast[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @bcast[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}
