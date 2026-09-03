//===- dma_to_channel_hoist_multi_arm_loop_anchor.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// An anchor spread over sibling arms, each inside its OWN LOOP, is replicated
// into every arm -- not lifted to the switch that holds them.
//
// The sibling test dma_to_channel_hoist_nested_arm_anchor.mlir establishes the
// opposite for arms with no loop in them: land ahead of the whole switch, since
// landing in one arm would starve the others. That stays right, and this does
// not contradict it. The distinction is the one the anchored placement already
// rests on elsewhere:
//
//   an ARM changes only WHETHER the transfer is issued -- context, and exactly
//   what an anchor is for, so inheriting it is correct;
//   a LOOP changes how MANY times -- a property of the transfer itself, which
//   must be neither inherited from a neighbour nor discarded.
//
// Climbing out of the arms here would cross both loops. The transfer issues the
// right number of TIMES either way, which is why the two look equivalent, but
// one lands inside the round loop and the other outside it. Outside, the hoist
// unrolls the transfer once per round at the outer scope: what should be one
// descriptor per round becomes trip-count-many, chained, and
// air-dependency-canonicalize rejects the result with 'arith.constant' op
// unknown op type. That is fused_decode's proj egress, whose two phases are the
// two arms and whose 36 and 46 round loops are these two scf.fors.
//
// Starving an arm is the failure of PICKING one. Taking all of them answers it
// rather than returning to it.

// CHECK-LABEL: func.func @multi_arm_loop_anchor
// One derived put per arm, each INSIDE that arm's own loop, immediately before
// the @w endpoint it anchored to.
// CHECK: scf.index_switch
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@t
// CHECK: air.channel.put{{.*}}@w
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@t
// CHECK: air.channel.put{{.*}}@w

air.channel @w [1]
air.channel @t [1]
func.func @multi_arm_loop_anchor(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %sel: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lb=%arg1, %s=%sel) : memref<64xbf16>, memref<64xbf16>, index {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c36 = arith.constant 36 : index
    %c46 = arith.constant 46 : index
    %c64 = arith.constant 64 : index
    // The two phases. Each runs its own round loop, and @w's only endpoints are
    // inside them.
    scf.index_switch %s
    case 0 {
      scf.for %r = %c0 to %c36 step %c1_l {
        air.channel.put @w[%c0] (%lb[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
      }
      scf.yield
    }
    default {
      scf.for %r = %c0 to %c46 step %c1_l {
        air.channel.put @w[%c0] (%lb[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
      }
      scf.yield
    }
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c1_s = arith.constant 1 : index
      // The transfer is in NEITHER arm: it sits plainly at segment scope, one
      // level shallower than every endpoint of its anchor.
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%sa) : memref<64xbf16> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c64_h = arith.constant 64 : index
        %l1 = memref.alloc() : memref<64xbf16, 2>
        air.dma_memcpy_nd (%l1[] [] [], %b[%c0_h] [%c64_h] [%c1_h]) {id = 1 : i32, channel = @t, channel_indices = array<i64: 0>, hoist_before = @w} : (memref<64xbf16, 2>, memref<64xbf16>)
        memref.dealloc %l1 : memref<64xbf16, 2>
      }
    }
  }
  return
}
