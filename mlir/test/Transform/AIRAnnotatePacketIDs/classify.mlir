//===- classify.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="report=true" -split-input-file -verify-diagnostics

// The volume discriminator: replication vs partition, and the cases that must
// NOT be mistaken for a demux.

// A true broadcast: each of the two destinations receives the WHOLE 512-element
// stream, so one routing id serves both.
module {
  // expected-remark @below {{broadcast over 2 destination(s); infers 1 routing id(s)}}
  air.channel @bcast [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
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

// -----

// A demux: the two destinations SPLIT the 1024-element stream (512 each), so
// each needs its own routing id for the switchbox to separate them.
module {
  // expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
  air.channel @demux [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @demux[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @demux[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @demux[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A partition on a channel that is NOT source-stamped cannot be a demux: with
// the DMA stamping one id on one BD there is no per-packet routing decision to
// make. Report it as unclassifiable rather than inventing ids.
module {
  // expected-remark @below {{not source-stamped}}
  air.channel @nohdr [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet"}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @nohdr[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @nohdr[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @nohdr[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A BUNDLE is not a demux. `air.channel @bundle [2]` is two independent
// parallel flows, not one flow fanning out; each is single-destination and
// needs exactly one id. Conflating the bundle index with a broadcast
// coordinate would report a spurious 2-way demux here.
module {
  // expected-remark @below {{single-destination over 1 destination(s); infers 1 routing id(s)}}
  air.channel @bundle [2] {channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %s0 = memref.alloc() : memref<512xbf16, 1 : i32>
    %s1 = memref.alloc() : memref<512xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @bundle[%c0] (%s0[] [] []) : (memref<512xbf16, 1 : i32>)
    air.channel.put @bundle[%c1] (%s1[] [] []) : (memref<512xbf16, 1 : i32>)
    air.channel.get @bundle[%c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @bundle[%c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// Header ownership is also implied by the ids themselves: a DMA can stamp at
// most ONE id, so several pinned ids mean the core must be writing the header.
// This is the shape air.channel's own docs use (packet_ids = [1,4] with no
// keep_pkt_header), and air-to-aie agrees via air::channelSourceWritesHeader.
// Treating it as DMA-stamped would skip demux classification entirely.
// expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
air.channel @ids_imply_header [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", packet_ids = [1 : i32, 4 : i32]}
func.func @f() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %src = memref.alloc() : memref<1024xbf16, 1 : i32>
  %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
  %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
  air.channel.put @ids_imply_header[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
  air.channel.get @ids_imply_header[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
  air.channel.get @ids_imply_header[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
  return
}

// -----

// A bundle that ALSO broadcasts: `[2,1]` fanning to `[2,2]` is two parallel
// instances each feeding two destinations. Only dimension 1 is the fan-out;
// dimension 0 selects the instance. Keying destinations on the whole index
// tuple would report 4 destinations here instead of 2 and misclassify.
// expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
air.channel @bundle_and_bcast [2, 1] {broadcast_shape = [2 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
func.func @g() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = memref.alloc() : memref<1024xbf16, 1 : i32>
  %s1 = memref.alloc() : memref<1024xbf16, 1 : i32>
  %a0 = memref.alloc() : memref<512xbf16, 2 : i32>
  %a1 = memref.alloc() : memref<512xbf16, 2 : i32>
  %b0 = memref.alloc() : memref<512xbf16, 2 : i32>
  %b1 = memref.alloc() : memref<512xbf16, 2 : i32>
  air.channel.put @bundle_and_bcast[%c0, %c0] (%s0[] [] []) : (memref<1024xbf16, 1 : i32>)
  air.channel.put @bundle_and_bcast[%c1, %c0] (%s1[] [] []) : (memref<1024xbf16, 1 : i32>)
  air.channel.get @bundle_and_bcast[%c0, %c0] (%a0[] [] []) : (memref<512xbf16, 2 : i32>)
  air.channel.get @bundle_and_bcast[%c0, %c1] (%a1[] [] []) : (memref<512xbf16, 2 : i32>)
  air.channel.get @bundle_and_bcast[%c1, %c0] (%b0[] [] []) : (memref<512xbf16, 2 : i32>)
  air.channel.get @bundle_and_bcast[%c1, %c1] (%b1[] [] []) : (memref<512xbf16, 2 : i32>)
  return
}
