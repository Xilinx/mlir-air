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

// The kernel stamps a DESTINATION ORDINAL and the pass substitutes the id it
// allocated for that destination. Ordinal k becomes ids[k], so the front end
// never names a wire number and the two spellings cannot drift apart.
//
// Substitution is BY VALUE, not by position: the arms below stamp 0, 1, 1 and
// the repeated ordinal must land on the repeated id. Relabelling positionally
// is what once sent one consumer's packets to another's.
//
// Note the arms are rewritten exactly once even though several calls trace
// back through the same switch -- rewriting during the walk let a later call
// read an earlier one's output and reject it as out of range.

// CHECK: air.channel @egress {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK-DAG: arith.constant 31 : i32
// CHECK-DAG: arith.constant 30 : i32
// CHECK-NOT: air.channel {{.*}}packet_ids = [0
module {
  air.channel @egress [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @stamp(memref<1024xbf16, 1 : i32>, i32)
  func.func @f(%phase: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ord0 = arith.constant 0 : i32
    %ord1 = arith.constant 1 : i32
    %src = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    // dest 0, then dest 1 twice -- the repeat is the point.
    %ord = scf.index_switch %phase -> i32
    case 0 {
      scf.yield %ord0 : i32
    }
    case 1 {
      scf.yield %ord1 : i32
    }
    default {
      scf.yield %ord1 : i32
    }
    func.call @stamp(%src, %ord) {air.pkt_header_channel = @egress, air.pkt_header_operand = 1 : i32} : (memref<1024xbf16, 1 : i32>, i32) -> ()
    air.channel.put @egress[%c0, %c0] (%src[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @egress[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}
