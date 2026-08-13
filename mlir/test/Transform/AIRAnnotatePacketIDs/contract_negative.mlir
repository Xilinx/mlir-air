//===- negative.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids -split-input-file -verify-diagnostics

// Cases the verifier must reject: the kernel-stamped id set and the channel's
// packet_ids are both statically known and disagree.

// A single stamped id that the channel does not route at all.
module {
  // expected-note @below {{channel declared here}}
  // expected-warning @below {{air-annotate-packet-ids disagrees with the packet_ids}}
  air.channel @outA [1] {channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [1 : i32, 4 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @f() {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %id = arith.constant 7 : i32
    // expected-error @below {{packet id contract violated: the kernel stamps [7] but channel 'outA' has [1, 4]}}
    // expected-note @below {{the switchbox routes only these ids}}
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @outA, air.pkt_header_operand = 1 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}

// -----

// A per-phase scf.index_switch whose arms stamp a set that is missing one of
// the pinned ids -- the id-8 destination would never receive a packet.
module {
  // expected-note @below {{channel declared here}}
  // expected-warning @below {{air-annotate-packet-ids disagrees with the packet_ids}}
  air.channel @outY [1] {channel_type = "npu_dma_packet", packet_ids = [1 : i32, 4 : i32, 8 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @f(%phase: index) {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %id = scf.index_switch %phase -> i32
    case 0 {
      scf.yield %c1 : i32
    }
    default {
      scf.yield %c4 : i32
    }
    // expected-error @below {{packet id contract violated: the kernel stamps [1, 4] but channel 'outY' has [1, 4, 8]}}
    // expected-note @below {{the switchbox routes only these ids}}
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @outY, air.pkt_header_operand = 1 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}

// -----

// Malformed marking: only one half of the pair is present.
module {
  air.channel @outA [1] {channel_type = "npu_dma_packet", packet_ids = [1 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @f() {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %id = arith.constant 1 : i32
    // expected-error @below {{must be set together}}
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @outA} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}

// -----

// Malformed marking: the operand index is out of range.
module {
  air.channel @outA [1] {channel_type = "npu_dma_packet", packet_ids = [1 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @f() {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %id = arith.constant 1 : i32
    // expected-error @below {{air.pkt_header_operand = 5 is out of range (call has 2 operands)}}
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @outA, air.pkt_header_operand = 5 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}

// -----

// Malformed marking: the symbol is not an air.channel.
module {
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @f() {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %id = arith.constant 1 : i32
    // expected-error @below {{references 'nosuch', which is not an air.channel in this module}}
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @nosuch, air.pkt_header_operand = 1 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}
