//===- dma_to_channel_hoist_outside_loops.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// air.hoist_outside_loops: resolve the anchor, then step out of the loops
// around it.
//
// An anchor names a channel, so it resolves to an op, so the transfer becomes
// that op's SIBLING and inherits its DEPTH exactly. When the transfer belongs
// one level shallower there has been no way to say so, and the failure is
// quiet: land a transfer inside a loop that its consumers sit outside of and
// the consumers stop dominating it -- "operand #N does not dominate this use".
//
// Stepping out of LOOPS and not out of ARMS is the distinction the whole hoist
// rests on. A loop changes how MANY times the transfer is issued, which is a
// property of the transfer and must never be inherited from a neighbour. An arm
// changes only WHETHER it is issued, which is a property of the surrounding
// context and is exactly what an anchor is for. Inherit the predicate, not the
// trip count.
//
// Here @a's only endpoint at the outer scope is inside BOTH a switch arm and an
// scf.for. The transfer belongs in the arm, once, ahead of the loop -- which is
// where the hand-written endpoint it replaces sat.

// CHECK-LABEL: func.func @outside_loops
// CHECK: scf.index_switch
// CHECK: default {
// The derived put is in the arm and AHEAD of the loop. Order is the whole
// assertion: without the directive it lands inside the loop, i.e. after it.
// CHECK: air.channel.put{{.*}}@t
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@a

air.channel @a [1]
air.channel @t [1]
func.func @outside_loops(%arg0: memref<64xbf16>, %sel: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %s=%sel) : memref<64xbf16>, index {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c2_l = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    scf.index_switch %s
    case 0 {
      scf.yield
    }
    default {
      // The anchor's endpoint is one loop deeper than the transfer belongs.
      scf.for %i = %c0 to %c2_l step %c1_l {
        air.channel.put @a[%c0] (%la[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
      }
      scf.yield
    }
    air.segment @seg args(%sa=%la, %ss=%s) : memref<64xbf16>, index {
      %c1_s = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%sa, %hs=%ss) : memref<64xbf16>, index {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c64_h = arith.constant 64 : index
        scf.index_switch %hs
        case 0 {
          scf.yield
        }
        default {
          %l1 = memref.alloc() : memref<64xbf16, 2>
          air.dma_memcpy_nd (%l1[] [] [], %b[%c0_h] [%c64_h] [%c1_h]) {id = 1 : i32, channel = @t, channel_indices = array<i64: 0>, hoist_before = @a, hoist_outside_loops} : (memref<64xbf16, 2>, memref<64xbf16>)
          memref.dealloc %l1 : memref<64xbf16, 2>
          scf.yield
        }
      }
    }
  }
  return
}
