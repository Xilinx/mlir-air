//===- nested_write_severs_forward.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="report=true" -split-input-file -verify-diagnostics

// Forwarding-edge recovery matches the buffer a get lands in against the buffer
// a later put sends. A write in between severs the link -- the payload was
// replaced, so nothing was forwarded. The sibling scan alone cannot see a write
// NESTED INSIDE the put's own enclosing op, which is the shape a compute core
// takes: land a buffer, then inside a loop fill it from a kernel and send it on.

// A COMPUTE CORE, NOT A RELAY. @up's payload lands in %buf; a call then
// overwrites %buf inside the loop that puts it on @dn. The two channels are
// independent, so @dn keeps its own domain and @up stays single-destination.
module {
  // expected-remark @below {{single-destination over 1 destination(s); infers 1 routing id(s)}}
  air.channel @up [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  // expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s)}}
  air.channel @dn [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @fill(memref<512xbf16, 2 : i32>)
  func.func @f(%n : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf = memref.alloc() : memref<512xbf16, 2 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 1 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 1 : i32>
    air.channel.get @up[%c0] (%buf[] [] []) : (memref<512xbf16, 2 : i32>)
    scf.for %i = %c0 to %n step %c1 {
      // The write the sibling scan cannot see: it is inside the loop, and the
      // loop is what sits between the get and the put at block level.
      func.call @fill(%buf) : (memref<512xbf16, 2 : i32>) -> ()
      air.channel.put @dn[%c0, %c0] (%buf[] [] []) dest(%c0) : (memref<512xbf16, 2 : i32>)
    }
    air.channel.put @dn[%c0, %c0] (%buf[] [] []) dest(%c1) : (memref<512xbf16, 2 : i32>)
    air.channel.get @dn[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 1 : i32>)
    air.channel.get @dn[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 1 : i32>)
    return
  }
}

// -----

// A GENUINE RELAY IS UNTOUCHED BY THE NESTING ITSELF. forward_chain.mlir's
// shape with the forwarding put moved inside a loop: same buffer, no call, so
// the payload really is handed onward and @hopA must still inherit @fanout's
// ids. This is what the nested scan must not sever.
module {
  // expected-remark @below {{infers 2 routing id(s) (forwarded from the demux @fanout)}}
  air.channel @hopA [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  // expected-remark @below {{demux over 2 destination(s); infers 2 routing id(s); fed by @hopA}}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @g() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hopA[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hopA[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    // The forwarding put, NESTED. One iteration, so the volumes are unchanged.
    scf.for %i = %c0 to %c1 step %c1 {
      air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    }
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}
