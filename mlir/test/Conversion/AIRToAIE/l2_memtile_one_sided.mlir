//===- l2_memtile_one_sided.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" -split-input-file -verify-diagnostics | FileCheck %s

// An L2 buffer read out to seven cores -- more MM2S channels than one memtile
// has, so specializeL2MemrefsIntoMemtiles takes it up for partitioning -- but
// written by no channel at all. Its list of gets is empty, and the
// non-overlap check over that list used to loop `i < size() - 1`, which wraps:
// the pass spun for 2^64 iterations.
//
// With no channel gets there is nothing to partition the buffer against.
// Seven circuit-switched flows cannot leave one memtile's six MM2S ports
// without two of them sharing a port, so this is rejected rather than lowered.

module {
  air.channel @to_core [7, 1]
  func.func @one_sided_l2() {
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
        // expected-error @+1 {{L2 memref is accessed by 7 circuit-switched channel puts, more than a memtile's 6 MM2S ports, and has no channel gets to partition it against}}
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

// The same buffer read out over a packet-switched channel is fine whole:
// the seven flows share one MM2S port, told apart by packet id.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.buffer(%{{.*}}) {{.*}} : memref<7x64xi32, 1>
// CHECK-COUNT-7: aie.packet_source<%{{.*}}, DMA : 0>
module {
  air.channel @to_core [7, 1] {channel_type = "npu_dma_packet"}
  func.func @one_sided_l2_packet() {
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
