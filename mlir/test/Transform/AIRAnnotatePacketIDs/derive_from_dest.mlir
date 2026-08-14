//===- derive_from_dest.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="assign=true emit-headers=false" -split-input-file | FileCheck %s

// A design states NOTHING about packet headers. Not keep_pkt_header, not
// air.src_writes_pkt_header, not packet_ids. Everything is derived from one
// fact the design already had to state: a put naming a `dest`.
//
// `dest(%d)` is a runtime value saying which leaf this packet is for. There is
// nothing else it can mean -- so the packet channel it reaches covers its
// broadcast dimension over TIME rather than over space, which is the entire
// space-vs-time question. The volume arithmetic (do the destinations partition
// the stream, or each consume all of it?) is evidence; the `dest` is the
// statement.
//
// This is the @outA -> @toMain -> @outY shape of the decode designs, reduced.

// The two hops MUST preserve the header: each carries a routing decision made
// upstream to a switchbox further downstream, and stripping it there would
// leave the demux nothing to route on. Forced by topology, so injected.
// CHECK-LABEL: @derived
// CHECK-DAG: air.channel @hopA {{.*}}keep_pkt_header, packet_ids = [31 : i32, 30 : i32, 29 : i32]
// CHECK-DAG: air.channel @hopB {{.*}}keep_pkt_header, packet_ids = [31 : i32, 30 : i32, 29 : i32]
// The demux gets the ids but NOT keep: at the final consumers, stripping to a
// pure payload is a real choice and stays the design's to make.
// CHECK-DAG: air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 3 : index], channel_type = "npu_dma_packet", packet_ids = [31 : i32, 30 : i32, 29 : i32]}
module @derived {
  air.channel @hopA [1] {channel_type = "npu_dma_packet"}
  air.channel @hopB [1] {channel_type = "npu_dma_packet"}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 3 : index], channel_type = "npu_dma_packet"}
  func.func @f(%d : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %src = memref.alloc() : memref<1536xbf16, 2 : i32>
    %mid = memref.alloc() : memref<1536xbf16, 1 : i32>
    %hub = memref.alloc() : memref<1536xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d2 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hopA[%c0] (%src[] [] []) dest(%d) : (memref<1536xbf16, 2 : i32>)
    air.channel.get @hopA[%c0] (%mid[] [] []) : (memref<1536xbf16, 1 : i32>)
    air.channel.put @hopB[%c0] (%mid[] [] []) : (memref<1536xbf16, 1 : i32>)
    air.channel.get @hopB[%c0] (%hub[] [] []) : (memref<1536xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1536xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c2] (%d2[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A channel that merely feeds the same buffer, without lying on a path from
// the originating put to the demux, is NOT a hop and gets nothing injected.
// The old attribute gate excluded such channels only incidentally; the path
// intersection excludes them on purpose.
// CHECK-LABEL: @offpath
// CHECK-DAG: air.channel @hop {{.*}}keep_pkt_header, packet_ids = [31 : i32, 30 : i32]
// Full attribute dict, so this also asserts the ABSENCE of keep and of ids --
// stronger than a CHECK-NOT, and immune to line ordering.
// CHECK-DAG: air.channel @aside [1] {channel_type = "npu_dma_packet"}
module @offpath {
  air.channel @hop [1] {channel_type = "npu_dma_packet"}
  air.channel @aside [1] {channel_type = "npu_dma_packet"}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet"}
  func.func @f(%d : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %other = memref.alloc() : memref<1024xbf16, 2 : i32>
    %sink = memref.alloc() : memref<1024xbf16, 1 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    // Unrelated flow: reaches nothing on the path to @fanout.
    air.channel.put @aside[%c0] (%other[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @aside[%c0] (%sink[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @hop[%c0] (%src[] [] []) dest(%d) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A design that predates `dest` and declares its own header ownership keeps
// working unchanged: with no dest anywhere, the walk falls back to the
// attribute-gated one.
// CHECK-LABEL: @legacy
// CHECK-DAG: air.channel @hop {{.*}}keep_pkt_header, packet_ids = [13 : i32, 17 : i32]
// CHECK-DAG: air.channel @fanout {{.*}}packet_ids = [13 : i32, 17 : i32]
module @legacy {
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
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
