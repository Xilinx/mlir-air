//===- dma_originator.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="assign=true" -split-input-file -verify-diagnostics | FileCheck %s

// The site that picks a packet's destination may be spelled as an
// air.dma_memcpy_nd, and the analysis has to see it.
//
// This pass runs BEFORE air-dma-to-channel, so a transfer written as a DMA
// naming a channel has no put and no get in the IR yet -- only the one op that
// will become both. The chain recovery is built out of puts and gets, so
// against a DMA-spelled producer it found nothing: no originator, and no
// forwarding edge either, since the edge is recovered by matching the buffer a
// get lands in against the buffer a later put sends. The demux was then
// rejected for a routing header "nothing upstream writes" -- while the header
// was in fact written exactly as before, by the same core, at the same offset.
//
// So a channel-naming DMA registers as a PAIR of endpoints. Which half is which
// follows from the direction of the copy and nothing else: the side it READS is
// the put, the side it WRITES is the get -- the same rule air-dma-to-channel
// applies when it materializes the two.

// The hop and the demux both come out with the same allocated id list, which is
// the whole point of a domain: a hop that admits fewer ids than the demux keys
// on filters the rest out at its own switchbox, and the demux never fires.
// CHECK-DAG: air.channel @hop {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK-DAG: air.channel @fan {{.*}}packet_ids = [31 : i32, 30 : i32]
// The header store lands on the DMA's SOURCE, at its source offset -- the
// switchbox reads the first word of what arrives, and the DMA sends the window
// starting there.
// CHECK: vector.bitcast
// CHECK: %[[OFF:.*]] = arith.constant 14 : index
// CHECK-NEXT: call @payload
// CHECK-NEXT: vector.store {{.*}}[%[[OFF]]] {alignment = 4 : i64} : memref<80xbf16, 2 : i32>, vector<2xbf16>
// CHECK: air.dma_memcpy_nd
// Once the word is in the payload the transfer is an ordinary one; `dest` is
// input to this pass, not a property carried onward.
// CHECK-NOT: dest(
module {
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @fan [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @payload(memref<80xbf16, 2 : i32>)
  func.func @f(%ph: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c66 = arith.constant 66 : index
    %b = memref.alloc() : memref<80xbf16, 2 : i32>
    %hub = memref.alloc() : memref<80xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<33xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<33xbf16, 2 : i32>
    %d = scf.index_switch %ph -> index
    case 0 {
      scf.yield %c0 : index
    }
    default {
      scf.yield %c1 : index
    }
    func.call @payload(%b) : (memref<80xbf16, 2 : i32>) -> ()
    air.dma_memcpy_nd (%hub[%c0] [%c66] [%c1], %b[%c14] [%c66] [%c1]) dest(%d) {channel = @hop} : (memref<80xbf16, 1 : i32>, memref<80xbf16, 2 : i32>)
    air.channel.put @fan[%c0, %c0] (%hub[] [] []) : (memref<80xbf16, 1 : i32>)
    air.channel.get @fan[%c0, %c0] (%d0[] [] []) : (memref<33xbf16, 2 : i32>)
    air.channel.get @fan[%c0, %c1] (%d1[] [] []) : (memref<33xbf16, 2 : i32>)
    return
  }
}

// -----

// NEGATIVE CONTROL for the above: the same IR with nothing claiming to write a
// header -- no `dest` on the DMA, and no `keep_pkt_header` asserting that some
// kernel stamps it by hand.
//
// Nothing then picks a destination, the L2 put on @fan forwards bytes it did
// not produce, and the design is the genuinely broken one the diagnostic exists
// for. It must still be rejected: making a DMA VISIBLE to the chain recovery
// must not become treating any DMA upstream as a header source. Note that the
// forwarding edge @hop -> @fan now exists in both modules -- it is registering
// the DMA's `dest` that separates them, not registering the DMA at all.
module {
  air.channel @hop [1] {channel_type = "npu_dma_packet"}
  // expected-error @+3 {{is a packet demux whose routing header nothing upstream writes}}
  // expected-note @+2 {{it forwards bytes it did not produce}}
  // expected-note @+1 {{no put naming a dest, and no forwarding hop, reaches it}}
  air.channel @fan [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @payload(memref<80xbf16, 2 : i32>)
  func.func @f(%ph: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c66 = arith.constant 66 : index
    %b = memref.alloc() : memref<80xbf16, 2 : i32>
    %hub = memref.alloc() : memref<80xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<33xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<33xbf16, 2 : i32>
    func.call @payload(%b) : (memref<80xbf16, 2 : i32>) -> ()
    air.dma_memcpy_nd (%hub[%c0] [%c66] [%c1], %b[%c14] [%c66] [%c1]) {channel = @hop} : (memref<80xbf16, 1 : i32>, memref<80xbf16, 2 : i32>)
    air.channel.put @fan[%c0, %c0] (%hub[] [] []) : (memref<80xbf16, 1 : i32>)
    air.channel.get @fan[%c0, %c0] (%d0[] [] []) : (memref<33xbf16, 2 : i32>)
    air.channel.get @fan[%c0, %c1] (%d1[] [] []) : (memref<33xbf16, 2 : i32>)
    return
  }
}

// -----

// Several producers gathering into one buffer, which is the shape the fused
// decode egress actually has: each core DMAs its own slice into the shared L2
// staging buffer, and one put forwards the whole thing.
//
// A DMA declares a write effect on its `dst`, so without an exemption the
// SECOND producer would look like a rewrite of the buffer between the first
// one's landing and the forwarding put, and sever the chain. It is not a
// rewrite -- it deposits bytes it did not transform, exactly as the several
// gets of an ordinary memtile gather do.
// CHECK-DAG: air.channel @ghop {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK-DAG: air.channel @gfan {{.*}}packet_ids = [31 : i32, 30 : i32]
module {
  air.channel @ghop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @gfan [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @payload(memref<80xbf16, 2 : i32>)
  func.func @f(%ph: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c66 = arith.constant 66 : index
    %b0 = memref.alloc() : memref<80xbf16, 2 : i32>
    %b1 = memref.alloc() : memref<80xbf16, 2 : i32>
    %hub = memref.alloc() : memref<160xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<66xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<66xbf16, 2 : i32>
    %d = scf.index_switch %ph -> index
    case 0 {
      scf.yield %c0 : index
    }
    default {
      scf.yield %c1 : index
    }
    func.call @payload(%b0) : (memref<80xbf16, 2 : i32>) -> ()
    func.call @payload(%b1) : (memref<80xbf16, 2 : i32>) -> ()
    air.dma_memcpy_nd (%hub[%c0] [%c66] [%c1], %b0[%c14] [%c66] [%c1]) dest(%d) {channel = @ghop} : (memref<160xbf16, 1 : i32>, memref<80xbf16, 2 : i32>)
    air.dma_memcpy_nd (%hub[%c66] [%c66] [%c1], %b1[%c14] [%c66] [%c1]) dest(%d) {channel = @ghop} : (memref<160xbf16, 1 : i32>, memref<80xbf16, 2 : i32>)
    air.channel.put @gfan[%c0, %c0] (%hub[] [] []) : (memref<160xbf16, 1 : i32>)
    air.channel.get @gfan[%c0, %c0] (%d0[] [] []) : (memref<66xbf16, 2 : i32>)
    air.channel.get @gfan[%c0, %c1] (%d1[] [] []) : (memref<66xbf16, 2 : i32>)
    return
  }
}
