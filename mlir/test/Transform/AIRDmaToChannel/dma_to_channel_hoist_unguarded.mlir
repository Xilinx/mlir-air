//===- dma_to_channel_hoist_unguarded.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// air.hoist_unguarded: place by default, but do not rebuild my guards.
//
// An unanchored hoist clones the guards the transfer sat under, so the external
// half stays conditional. An anchored one skips them and inherits the anchor's
// control context instead. A transfer whose hand-written counterpart was
// UNGUARDED at the outer scope wants neither.
//
// Not all guards are equal. One on the hierarchy's own induction variable is
// harmless -- the tile-count machinery collapses it and nothing is rebuilt. One
// on an i32 RUNTIME PARAMETER, as here, is not: the rebuild emits
// `arith.index_cast` on the SEGMENT'S OWN i32 block argument, and that does not
// survive the segment becoming an aie.device. air-to-aie rejects it with
// "'arith.index_cast' op using value defined outside the region".
//
// Anchoring is the other way to skip the rebuild, but it needs an endpoint at
// the right control depth to name. In fused_decode's hybrid there is none: the
// preceding segment-scope endpoint sits inside both a switch arm and an
// scf.for, and the following one is already anchored back to this transfer.
//
// Note the two halves must part company here. The INTERNAL half stays inside
// the arm -- the core really does only run on one arm. It is the EXTERNAL half
// whose hand-written counterpart was unconditional.

// CHECK-LABEL: func.func @rtp_guard
// The external half lands at segment scope, bare: no rebuilt index_switch and
// no index_cast of the segment's i32 argument.
// CHECK: air.segment
// CHECK-NOT: arith.index_cast
// CHECK-NOT: scf.index_switch
// CHECK: air.channel.get{{.*}}@t
// The internal half keeps its arm.
// CHECK: air.herd
// CHECK: scf.index_switch
// CHECK: air.channel.put{{.*}}@t

air.channel @t [1]
func.func @rtp_guard(%arg0: memref<64xbf16>, %rt: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lrt=%rt) : memref<64xbf16>, i32 {
    air.segment @seg args(%sa=%la, %srt=%lrt) : memref<64xbf16>, i32 {
      %c1_s = arith.constant 1 : index
      %l2 = memref.alloc() : memref<64xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%d=%l2, %hrt=%srt) : memref<64xbf16, 1>, i32 {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c64_h = arith.constant 64 : index
        // The arm: a per-layer runtime parameter, not a constant. This is the
        // guard that cannot be rebuilt outside the herd.
        %i = arith.index_cast %hrt : i32 to index
        scf.index_switch %i
        case 0 {
          scf.yield
        }
        default {
          %l1 = memref.alloc() : memref<64xbf16, 2>
          air.dma_memcpy_nd (%d[%c0_h] [%c64_h] [%c1_h], %l1[] [] []) {id = 1 : i32, channel = @t, channel_indices = array<i64: 0>, hoist_unguarded} : (memref<64xbf16, 1>, memref<64xbf16, 2>)
          memref.dealloc %l1 : memref<64xbf16, 2>
          scf.yield
        }
      }
      memref.dealloc %l2 : memref<64xbf16, 1>
    }
  }
  return
}
