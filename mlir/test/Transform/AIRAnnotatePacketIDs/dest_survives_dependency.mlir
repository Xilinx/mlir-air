//===- dest_survives_dependency.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-annotate-packet-ids="assign=true" | FileCheck %s

// The header store is emitted AFTER air-dependency, so `dest` has to survive
// it. air-dependency re-instantiates every channel op with an async token, and
// that builder does not take the optional operand: it has to be carried over
// explicitly. When it was not, the pass simply found nothing to do -- no error,
// no header written, and a design that routes packets to whichever destination
// the switchbox defaults to.
//
// Emitting after air-dependency is not a preference. The store must share a
// lock section with the payload write, which means it must go INSIDE that
// write's air.execute -- a region that does not exist until air-dependency
// creates it. Emitting earlier gets the store its own region, its own lock
// section, and a buffer handed to the DMA before the header is in it.

// CHECK: air.channel @egress {{.*}}packet_ids = [31 : i32, 30 : i32]
// The store is in the same region as the payload write, not beside it: were it
// beside, an air.execute_terminator would separate the two.
// CHECK: call @payload
// CHECK-NEXT: vector.store {{.*}}{alignment = 4 : i64} : memref<80xbf16, 2 : i32>, vector<2xbf16>
// CHECK: air.channel.put
// The operand is consumed, not left behind for later stages to trip over.
// CHECK-NOT: dest(

module {
  air.channel @egress [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func private @payload(memref<80xbf16, 2 : i32>)
  func.func @f(%ph: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c14 = arith.constant 14 : index
    %c66 = arith.constant 66 : index
    %b = memref.alloc() : memref<80xbf16, 2 : i32>
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
    air.channel.put @egress[%c0, %c0] (%b[%c14] [%c66] [%c1]) dest(%d) : (memref<80xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c0] (%d0[] [] []) : (memref<33xbf16, 2 : i32>)
    air.channel.get @egress[%c0, %c1] (%d1[] [] []) : (memref<33xbf16, 2 : i32>)
    return
  }
}
