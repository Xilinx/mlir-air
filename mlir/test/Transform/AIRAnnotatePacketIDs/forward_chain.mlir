//===- forward_chain.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="report=true" -split-input-file -verify-diagnostics

// Back-propagation: a single-destination hop that FORWARDS packets a
// downstream demux will route must admit every id the demux keys on, or the
// extra ids are filtered out mid-route and the demux never fires. This is the
// @outA -> @toMain -> @outY shape of the decode designs, reduced.

module {
  // Two hops upstream of the demux. Each is single-destination on its own, but
  // both must carry the demux's 2 ids.
  // expected-remark @below {{infers 2 routing id(s) (forwarded from the demux @fanout)}}
  air.channel @hopA [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  // expected-remark @below {{infers 2 routing id(s) (forwarded from the demux @fanout)}}
  air.channel @hopB [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  // expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}

  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %mid = memref.alloc() : memref<1024xbf16, 1 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>

    // core L1 -> column memtile
    air.channel.put @hopA[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hopA[%c0] (%mid[] [] []) : (memref<1024xbf16, 1 : i32>)
    // column memtile -> hub memtile (same memref hands the chain onward)
    air.channel.put @hopB[%c0] (%mid[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @hopB[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    // hub -> the id-demux egress
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A hop that is NOT header-preserving re-stamps and starts a fresh routing
// domain, so demand does not propagate through it: it keeps its single id even
// though it feeds the same demux.
module {
  // expected-remark @below {{single-destination over 1 destination(s); infers 1 routing id(s)}}
  air.channel @restamp [1] {channel_type = "npu_dma_packet"}
  // expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}

  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @restamp[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @restamp[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// Cross-check against a pin: a working design's packet_ids is ground truth for
// the inference, so a disagreement is reported rather than silently resolved.
// Here the channel pins 3 ids but only 2 destinations exist.
module {
  // expected-remark @below {{demux over 2 destination(s)}}
  // expected-warning @below {{air-annotate-packet-ids disagrees with the packet_ids on @fanout: inferred 2 id(s)}}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [1 : i32, 4 : i32, 8 : i32]}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @fanout[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}
