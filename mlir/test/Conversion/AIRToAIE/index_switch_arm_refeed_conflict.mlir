//===- index_switch_arm_refeed_conflict.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" 2>&1 | FileCheck %s

// Two mutually-exclusive scf.index_switch arms consume the SAME air.channel into
// per-arm buffers that ask for DIFFERENT re-send counts. unifyIndexSwitchArmBuffers
// folds those buffers onto one shared buffer -- correct, and what keeps the idle
// arm's dma_start from ahead-blocking the active one -- but only ONE count can
// ride the result.
//
// It used to keep whichever arm it visited first and drop the other silently.
// That is the worst possible outcome: the count is a lock value baked into a DMA
// BD, the arms share one BD chain, and the chain advances one BD per completed
// transfer with no notion of arms. The arm whose count was dropped gets no replay
// at all, so the build is clean, the IR looks converted, and the device hangs on
// whichever dispatch first takes that arm.
//
// Say so instead.

// CHECK: error:
// CHECK-SAME: arms share air.channel @fill
// CHECK-SAME: different air.refeed_count
// CHECK-SAME: 12
// CHECK-SAME: 30
// CHECK-SAME: cannot branch

air.channel @fill [1, 1]
func.func @arm_refeed_conflict(%ext: memref<32xbf16>, %mode: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext, %m=%mode)
        : memref<32xbf16>, i32 {
    air.channel.put @fill[] (%e[] [] []) : (memref<32xbf16>)
    air.segment @seg args(%sm=%m) : i32 {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%hm=%sm) : i32
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %i = arith.index_cast %hm : i32 to index
        scf.index_switch %i
        case 0 {
          %a = memref.alloc() {air.refeed_count = 12 : i32} : memref<32xbf16, 2>
          air.channel.get @fill[] (%a[] [] []) : (memref<32xbf16, 2>)
          memref.dealloc %a : memref<32xbf16, 2>
          scf.yield
        }
        default {
          %b = memref.alloc() {air.refeed_count = 30 : i32} : memref<32xbf16, 2>
          air.channel.get @fill[] (%b[] [] []) : (memref<32xbf16, 2>)
          memref.dealloc %b : memref<32xbf16, 2>
          scf.yield
        }
      }
    }
  }
  return
}
