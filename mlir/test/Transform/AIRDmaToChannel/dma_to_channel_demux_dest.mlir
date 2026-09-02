//===- dma_to_channel_demux_dest.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// A runtime packet-demux destination has to reach the derived air.channel.put.
//
// `dest` selects, at run time, which consumer of a demux a transfer is for; the
// compiler allocates that destination's packet id and emits the routing-header
// store. It is an OPERAND, not an attribute, so a transfer that needs one could
// not be spelled as an air.dma_memcpy_nd at all -- the op had nowhere to put it.
// Every gather whose producers pick their consumer at run time was therefore
// stuck on the hand-written put/get pair.
//
// It rides on the INTERNAL half, the one that stays on the core: that is where
// the demux index is computed and what air-annotate-packet-ids reads. The
// external half is the memtile or shim side and has no destination to select.

// CHECK-LABEL: func.func @demux
// CHECK: air.herd
// CHECK: air.channel.put{{.*}}@a{{.*}}dest(

air.channel @a [2, 2] {channel_type = "npu_dma_packet"}
func.func @demux(%arg0: memref<256xbf16, 1>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %l2 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%d=%l2) : memref<256xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c64_h = arith.constant 64 : index
        %sel = arith.addi %tx, %ty : index
        %l1 = memref.alloc() : memref<64xbf16, 2>
        air.dma_memcpy_nd (%d[%c0_h] [%c64_h] [%c1_h], %l1[] [] []) dest(%sel) {id = 1 : i32, channel = @a, channel_indices = array<i64: 0, 0>} : (memref<256xbf16, 1>, memref<64xbf16, 2>)
        memref.dealloc %l1 : memref<64xbf16, 2>
      }
      memref.dealloc %l2 : memref<256xbf16, 1>
    }
  }
  return
}
