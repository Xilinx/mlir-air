//===- positive.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids -split-input-file -verify-diagnostics | FileCheck %s

// Cases the verifier must accept: either the two spellings agree, or the
// contract is not statically checkable and the pass stays silent.

// The real decode shape: a per-phase index_switch stamping exactly the pinned
// set. Which arm fires is a runtime property; the set is what must match.
// CHECK-LABEL: @agrees
module {
  // expected-warning @below {{air-annotate-packet-ids disagrees with the packet_ids}}
  air.channel @outA [1] {channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [1 : i32, 4 : i32, 8 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @agrees(%phase: index) {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %id = scf.index_switch %phase -> i32
    case 0 {
      scf.yield %c1 : i32
    }
    case 1 {
      scf.yield %c4 : i32
    }
    case 2 {
      scf.yield %c8 : i32
    }
    default {
      // The down phase reuses the o-proj id -- a repeat, not a fourth id.
      scf.yield %c4 : i32
    }
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @outA, air.pkt_header_operand = 1 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}

// -----

// Unresolvable operand (loop-carried): unchecked, not violated. Silent.
// CHECK-LABEL: @dynamic_id_is_silent
module {
  // expected-warning @below {{air-annotate-packet-ids disagrees with the packet_ids}}
  air.channel @outA [1] {channel_type = "npu_dma_packet", packet_ids = [1 : i32, 4 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @dynamic_id_is_silent(%n: index) {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %seed = arith.constant 1 : i32
    %step = arith.constant 3 : i32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %seed) -> (i32) {
      func.call @stamp(%buf, %acc) {air.pkt_header_channel = @outA, air.pkt_header_operand = 1 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
      %next = arith.addi %acc, %step : i32
      scf.yield %next : i32
    }
    return
  }
}

// -----

// Channel pins nothing: there is no second spelling to disagree with. Silent.
// CHECK-LABEL: @unpinned_is_silent
module {
  air.channel @outA [1] {channel_type = "npu_dma_packet"}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @unpinned_is_silent() {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %id = arith.constant 7 : i32
    func.call @stamp(%buf, %id) {air.pkt_header_channel = @outA, air.pkt_header_operand = 1 : i32} : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}

// -----

// Unmarked calls are ignored entirely, pinned channel or not.
// CHECK-LABEL: @unmarked_is_ignored
module {
  air.channel @outA [1] {channel_type = "npu_dma_packet", packet_ids = [1 : i32]}
  func.func private @stamp(memref<48xbf16, 2 : i32>, i32)
  func.func @unmarked_is_ignored() {
    %buf = memref.alloc() : memref<48xbf16, 2 : i32>
    %id = arith.constant 7 : i32
    func.call @stamp(%buf, %id) : (memref<48xbf16, 2 : i32>, i32) -> ()
    return
  }
}
