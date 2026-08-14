//===- no_header_source.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="assign=true emit-headers=false" -split-input-file -verify-diagnostics

// The faults below all used to compile clean and hang the device. A packet
// demux routes on a header word that is part of the payload, so SOMETHING
// upstream has to have written it. When the chain that would have carried that
// header is severed, every channel still looks well-formed on its own -- the
// hop just quietly keeps its single id, its switchbox filters the rest out
// mid-route, and the demux never fires.

// A compute between the get and the put is a TRANSFORM, not a forward: it
// overwrites the payload the get landed, header and all. So @hop is correctly
// NOT a hop -- and that leaves the demux with no header source at all, which
// is the error worth reporting.
module {
  func.func private @compute(memref<1024xbf16, 1 : i32>)
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  // expected-error @below {{is a packet demux whose routing header nothing upstream writes}}
  // expected-note @below {{its put reads an L2/L3 buffer}}
  // expected-note @below {{no put naming a dest, and no forwarding hop, reaches it}}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hop[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    func.call @compute(%hub) : (memref<1024xbf16, 1 : i32>) -> ()
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A put that names a destination, with no demux anywhere in its domain. The
// diagnostic names the DOMAIN: the old one could only blame the put's own
// channel for "not being a demux", which is misleading when that channel is a
// hop and the real fault is a severed link downstream.
module {
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    // expected-error @below {{selects a destination, but its routing domain has no demux}}
    // expected-note @below {{reaches no packet channel with more than one destination}}
    // expected-note @below {{no packet channel forwards @hop's payload onward}}
    air.channel.put @hop[%c0] (%src[] [] []) dest(%c0) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    return
  }
}

// -----

// A demux fed only from L2 with nothing upstream. A memtile is a pure data
// mover: it forwards bytes it did not produce and cannot have chosen where
// they go -- which is the entire reason the decision travels in the header
// rather than being baked into a BD.
module {
  // expected-error @below {{is a packet demux whose routing header nothing upstream writes}}
  // expected-note @below {{its put reads an L2/L3 buffer}}
  // expected-note @below {{no put naming a dest, and no forwarding hop, reaches it}}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
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

// -----

// ...but a demux whose put comes from L1 is fine with no upstream: a compute
// core CAN have written the header itself. No diagnostic.
module {
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @fanout[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// ...and so is a demux that DECLARES its ids. An explicit list is the design
// asserting it knows its own routing, possibly via a header source this
// analysis cannot see. The fault above is about a list the compiler DERIVED,
// where a wrong derivation is the compiler's fault to report.
module {
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [13 : i32, 17 : i32]}
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

// -----

// A hop pinning the domain's ids in a DIFFERENT ORDER is fine. A hop is
// single-destination, so air-to-aie returns its whole list for the one buffer
// and the sequence carries no meaning. No diagnostic.
module {
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [17 : i32, 13 : i32]}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [13 : i32, 17 : i32]}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hop[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A hop pinning the RIGHT NUMBER of ids and the WRONG ONES. Comparing counts
// alone would wave this through, and the design would hang: @hop's switchbox
// admits 1 and 2, the demux keys on 13 and 17, so every packet is filtered out
// at the hop and the demux never fires.
module {
  // expected-error @below {{pins routing ids [1, 2], but forwards for a demux routing [13, 17]}}
  // expected-note @below {{the demux is @fanout}}
  // expected-note @below {{a hop must admit exactly the ids the demux keys on}}
  // expected-note @below {{the order of a hop's list is not checked}}
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [1 : i32, 2 : i32]}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [13 : i32, 17 : i32]}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hop[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}
