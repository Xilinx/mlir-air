//===- assign.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="assign=true" -split-input-file | FileCheck %s

// assign mode COPIES a declared id list onto the hops that forward its packets.
// It never invents a value and never rewrites a kernel constant -- both were
// tried and both broke llama-3b on device. Inventing renumbers every other
// packet channel (air-to-aie hands out the lowest free ids) and perturbs a
// tuned floorplan. Relabeling positionally assumes the kernel's arm order
// matches the declared list, which on llama swaps 1 and 4 and routes rope's
// packets to rms.
//
// The demux declares, because its list is ORDERED: air-to-aie routes
// destination i with packet_ids[i]. A forwarding hop is single-destination, so
// air-to-aie hands it the whole list for one buffer -- its order carries no
// meaning and its set is just the demux's. That half is derivable, so the front
// end does not have to write it.
// CHECK-DAG: air.channel @hopA {{.*}}packet_ids = [7 : i32, 9 : i32]
// CHECK-DAG: air.channel @fanout {{.*}}packet_ids = [7 : i32, 9 : i32]
// The kernel constants are left exactly as written.
// CHECK-DAG: arith.constant 7 : i32
// CHECK-DAG: arith.constant 9 : i32
module {
  air.channel @hopA [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [7 : i32, 9 : i32]}
  func.func private @stamp(memref<1024xbf16, 2 : i32>, i32)
  func.func @f(%phase: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>

    %c7 = arith.constant 7 : i32
    %c9 = arith.constant 9 : i32
    %id = scf.index_switch %phase -> i32
    case 0 {
      scf.yield %c7 : i32
    }
    default {
      scf.yield %c9 : i32
    }
    func.call @stamp(%src, %id) {air.pkt_header_channel = @hopA, air.pkt_header_operand = 1 : i32} : (memref<1024xbf16, 2 : i32>, i32) -> ()

    air.channel.put @hopA[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hopA[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A demux that declares nothing gets its ids ALLOCATED, one per destination,
// counted from the top of the id space. Taking them downward from kMaxPacketID
// keeps them clear of the upward auto-assignment air-to-aie gives every other
// packet flow, so adding a demux does not renumber the rest of the design --
// claiming the low end instead is what broke llama-3b on device.
// CHECK: air.channel @unmarked {{.*}}packet_ids = [31 : i32, 30 : i32]
module {
  air.channel @unmarked [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @unmarked[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @unmarked[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @unmarked[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// An explicit pin is an OVERRIDE: assign mode must leave it alone.
// CHECK: air.channel @pinned {{.*}}packet_ids = [13 : i32, 17 : i32]
module {
  air.channel @pinned [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [13 : i32, 17 : i32]}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @pinned[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @pinned[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @pinned[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A plain broadcast needs no pin at all -- one id, which the existing
// air-to-aie allocator already supplies. assign mode must not add one.
// CHECK-NOT: packet_ids
module {
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

// The put names its destination; the compiler writes the routing header.
//
// `dest` is the index, along the broadcast dimension, of the consumer this
// packet is for -- the same index the matching get sits at. From that alone the
// pass allocates the ids, maps the runtime destination to one, and emits the
// store. Where the header goes is not a choice: the switchbox reads the first
// word of what arrives and the DMA sends from offsets[0], so the header IS
// offsets[0] -- element 14 here.
//
// This used to be a hand-written kernel call taking the id as an argument, with
// the same ids also spelled on the channel. Two spellings, no link, and a
// silent misroute when they disagreed.
//
// WHERE the store goes matters as much as what it writes. The buffer is
// lock-protected -- producer acquires, writes, releases to the consumer -- and
// a write outside that section gets a section of its own: the buffer reaches
// the DMA with no header, and the consumer lock is signalled twice per
// production. On device that is a hang. So the store lands immediately after
// the payload write, INSIDE its section, ahead of the put.
//
// Only the store follows the payload write; the arithmetic picking the id goes
// ahead of it, where it costs nothing (it touches no buffer). When the payload
// write sits in an air.execute -- the usual case, see the next test -- the
// store goes inside that region instead.
//
// Either way the store cannot reuse the put's offset Values: %c14 here is
// materialized after the payload call, so referring to it from the anchor is a
// dominance violation. Constant offsets are re-materialized instead.

// CHECK: air.channel @egress {{.*}}packet_ids = [31 : i32, 30 : i32]
// The ids allocated here are a descending run, so the lookup is affine and
// collapses to one multiply-add. A non-affine pin falls back to a select chain.
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.index_cast
// CHECK: vector.broadcast
// CHECK: vector.bitcast {{.*}}vector<1xi32> to vector<2xbf16>
// CHECK: %[[OFF:.*]] = arith.constant 14 : index
// CHECK: call @payload
// CHECK-NEXT: vector.store {{.*}}[%[[OFF]]] {alignment = 4 : i64} : memref<80xbf16, 2 : i32>, vector<2xbf16>
// CHECK: air.channel.put{{ *}}@egress
module {
  air.channel @egress [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @payload(memref<80xbf16, 2 : i32>)
  func.func @f(%ph: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b  = memref.alloc() : memref<80xbf16, 2 : i32>
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
    %c14 = arith.constant 14 : index
    %c66 = arith.constant 66 : index
    air.channel.put @egress[%c0, %c0] (%b[%c14] [%c66] [%c1]) dest(%d) : (memref<80xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c0] (%d0[] [] []) : (memref<33xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c1] (%d1[] [] []) : (memref<33xbf16, 2 : i32>)
    return
  }
}

// -----

// The payload write is normally wrapped in an air.execute, and then the header
// store must go INSIDE that region -- not merely next to it.
//
// Adjacency is not enough. A store emitted beside the execute becomes an
// execute of its own, and the lock placer brackets each one separately: the
// buffer reaches the DMA after the payload with no header, and the consumer
// lock is signalled twice per production. On device that is a hang. One region,
// one lock section -- the same shape the hand-written kernel had when it
// stamped the header itself.

// CHECK: air.execute
// CHECK-NEXT: call @payload
// CHECK-NEXT: vector.store {{.*}}{alignment = 4 : i64}
module {
  air.channel @egress [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @payload(memref<80xbf16, 2 : i32>)
  func.func @f(%ph: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b  = memref.alloc() : memref<80xbf16, 2 : i32>
    %d0 = memref.alloc() : memref<33xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<33xbf16, 2 : i32>
    %d = scf.index_switch %ph -> index
    case 0 {
      scf.yield %c0 : index
    }
    default {
      scf.yield %c1 : index
    }
    %t = air.execute {
      func.call @payload(%b) : (memref<80xbf16, 2 : i32>) -> ()
    }
    %c14 = arith.constant 14 : index
    %c66 = arith.constant 66 : index
    air.channel.put @egress[%c0, %c0] (%b[%c14] [%c66] [%c1]) dest(%d) : (memref<80xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c0] (%d0[] [] []) : (memref<33xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c1] (%d1[] [] []) : (memref<33xbf16, 2 : i32>)
    return
  }
}
