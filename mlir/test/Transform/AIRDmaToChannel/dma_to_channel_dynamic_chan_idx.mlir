//===- dma_to_channel_dynamic_chan_idx.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// A sub-channel selector that is only known at run time.
//
// `channel_indices` is a static DenseI64ArrayAttr, but air.channel.put/get take
// their `indices` as OPERANDS. A transfer that selects its sub-channel with a
// value -- a tile indexing its own column, `base + tx` -- therefore had no
// spelling as an air.dma_memcpy_nd at all, and every such gather was stuck on
// the hand-written pair. That is what holds fused_decode's @outA.
//
// Runtime indices win over the static attribute and over the spatial inference:
// the front end has said exactly which sub-channel this is, and it is not a
// constant. Both halves index the same one.

// CHECK-LABEL: func.func @dynidx
// Both halves take the herd's induction variables as their index, not a
// materialized constant.
// ONE index, the computed selector -- not the two herd induction variables the
// spatial inference would have supplied.
// CHECK: air.channel.get{{.*}}@a[%{{[0-9]+}}]
// CHECK-NOT: air.channel.get{{.*}}@a[%arg{{[0-9]+}}, %arg{{[0-9]+}}]
// CHECK: air.channel.put{{.*}}@a[%{{[0-9]+}}]

air.channel @a [4] {channel_type = "npu_dma_packet"}
func.func @dynidx(%arg0: memref<256xbf16, 1>) {
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
        air.dma_memcpy_nd (%d[%c0_h] [%c64_h] [%c1_h], %l1[] [] []) dest(%sel) chan_idx(%sel) {id = 1 : i32, channel = @a} : (memref<256xbf16, 1>, memref<64xbf16, 2>)
        memref.dealloc %l1 : memref<64xbf16, 2>
      }
      memref.dealloc %l2 : memref<256xbf16, 1>
    }
  }
  return
}
