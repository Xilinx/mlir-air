//===- dma_port_sharing.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" -split-input-file -verify-diagnostics

// Once every DMA channel is allocated, air-to-aie rejects a port shared in a
// way the switchbox cannot carry. A tile out of free channels gets a flow
// wrapped onto a channel already in use; these are the cases where that
// joined two flows that cannot share it. Every case but the last puts more
// flows on one memtile than it has DMA channels on that side (6); the last
// pins two flows onto one port.

// Seven circuit-switched flows out of one memtile: the seventh shares MM2S 0
// with the first, and both cores would receive both flows' data.

module {
  // expected-error @+2 {{MM2S channel 0 streams to different destinations for two circuit-switched flows}}
  // expected-note @+1 {{the other flow}}
  air.channel @to_core [7, 1]
  func.func @circuit_puts() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @seg0 {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c64 = arith.constant 64 : index
        %l2 = memref.alloc() : memref<7x64xi32, 1>
        air.channel.put @to_core[%c0, %c0] (%l2[%c0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c1_0, %c0] (%l2[%c1_0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c2, %c0] (%l2[%c2, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c3, %c0] (%l2[%c3, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c4, %c0] (%l2[%c4, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c5, %c0] (%l2[%c5, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c6, %c0] (%l2[%c6, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<7x64xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c7, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<7x64xi32, 1>
      }
    }
    return
  }
}

// -----

// The receiving side: seven circuit-switched flows into one memtile, two of
// them on S2MM 0.

module {
  // expected-error @+2 {{S2MM channel 0 is fed by two circuit-switched sources}}
  // expected-note @+1 {{the other flow}}
  air.channel @from_core [7, 1]
  func.func @circuit_gets() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @seg0 {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c64 = arith.constant 64 : index
        %l2 = memref.alloc() : memref<7x64xi32, 1>
        air.channel.get @from_core[%c0, %c0] (%l2[%c0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<7x64xi32, 1>)
        air.channel.get @from_core[%c1_0, %c0] (%l2[%c1_0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<7x64xi32, 1>)
        air.channel.get @from_core[%c2, %c0] (%l2[%c2, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<7x64xi32, 1>)
        air.channel.get @from_core[%c3, %c0] (%l2[%c3, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<7x64xi32, 1>)
        air.channel.get @from_core[%c4, %c0] (%l2[%c4, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<7x64xi32, 1>)
        air.channel.get @from_core[%c5, %c0] (%l2[%c5, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<7x64xi32, 1>)
        air.channel.get @from_core[%c6, %c0] (%l2[%c6, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<7x64xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c7, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.put @from_core[%tx, %ty] (%buf[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<7x64xi32, 1>
      }
    }
    return
  }
}

// -----

// Six circuit-switched flows fill the six MM2S channels, and a packet-switched
// flow is put on MM2S 0 beside a circuit one. A switchbox port is one or the
// other.

module {
  // expected-note @+1 {{the other flow}}
  air.channel @to_core [6, 1]
  // expected-error @+1 {{MM2S channel 0 carries both a packet-switched and a circuit-switched flow}}
  air.channel @to_core_pkt [1, 1] {channel_type = "npu_dma_packet"}
  func.func @mixed_kinds() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @seg0 {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c64 = arith.constant 64 : index
        %l2 = memref.alloc() : memref<7x64xi32, 1>
        air.channel.put @to_core[%c0, %c0] (%l2[%c0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c1_0, %c0] (%l2[%c1_0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c2, %c0] (%l2[%c2, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c3, %c0] (%l2[%c3, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c4, %c0] (%l2[%c4, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c5, %c0] (%l2[%c5, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core_pkt[%c0, %c0] (%l2[%c6, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<7x64xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c6, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        air.herd @herd_1 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @to_core_pkt[%tx, %ty] (%buf[] [] []) {id = 9 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<7x64xi32, 1>
      }
    }
    return
  }
}

// -----

// The memtile buffer is also written by a channel, but the reads overlap, so
// it is not partitioned and keeps all seven circuit-switched reads.

module {
  // expected-error @+2 {{MM2S channel 0 streams to different destinations for two circuit-switched flows}}
  // expected-note @+1 {{the other flow}}
  air.channel @to_core [7, 1]
  air.channel @fill [1, 1]
  func.func @two_sided_overlap(%in: memref<7x64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%a=%in) : memref<7x64xi32> {
      air.channel.put @fill[] (%a[] [] []) {id = 20 : i32} : (memref<7x64xi32>)
      air.segment @seg0 {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %c64 = arith.constant 64 : index
        %l2 = memref.alloc() : memref<7x64xi32, 1>
        air.channel.get @fill[] (%l2[] [] []) {id = 21 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c0, %c0] (%l2[%c0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c1_0, %c0] (%l2[%c1_0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 2 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c2, %c0] (%l2[%c2, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 3 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c3, %c0] (%l2[%c3, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 4 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c4, %c0] (%l2[%c4, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 5 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c5, %c0] (%l2[%c5, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 6 : i32} : (memref<7x64xi32, 1>)
        air.channel.put @to_core[%c6, %c0] (%l2[%c0, %c0] [%c1_0, %c64] [%c64, %c1_0]) {id = 7 : i32} : (memref<7x64xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c7, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<7x64xi32, 1>
      }
    }
    return
  }
}

// -----

// air.tile_dma_channel cannot force a packet- and a circuit-switched flow onto
// one port: the switchbox still cannot carry it, so the pin is rejected rather
// than left for aie.device's verifier.

module {
  // expected-note @+1 {{the other flow}}
  air.channel @q4 [1] {air.tile_dma_channel = 0 : i32}
  // expected-error @+1 {{carries both a packet-switched and a circuit-switched flow}}
  air.channel @k4 [1] {air.tile_dma_channel = 0 : i32, channel_type = "npu_dma_packet"}
  func.func @pinned_mixed_kinds(%ext: memref<8xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
      air.segment @seg4 args(%se=%e) : memref<8xbf16> {
        %c1_0 = arith.constant 1 : index
        air.herd @h4 tile(%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %bq = memref.alloc() : memref<8xbf16, 2>
          %bk = memref.alloc() : memref<8xbf16, 2>
          air.channel.put @q4[%tx, %ty] (%bq[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
          air.channel.put @k4[%tx, %ty] (%bk[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
          memref.dealloc %bq : memref<8xbf16, 2>
          memref.dealloc %bk : memref<8xbf16, 2>
        }
        %lq = memref.alloc() : memref<8xbf16, 1>
        %lk = memref.alloc() : memref<8xbf16, 1>
        air.channel.get @q4[] (%lq[] [] []) {id = 3 : i32} : (memref<8xbf16, 1>)
        air.channel.get @k4[] (%lk[] [] []) {id = 4 : i32} : (memref<8xbf16, 1>)
        memref.dealloc %lq : memref<8xbf16, 1>
        memref.dealloc %lk : memref<8xbf16, 1>
      }
    }
    return
  }
}
