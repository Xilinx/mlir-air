//===- dma_to_channel_pure_guard_def.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// A guard rebuilt by an earlier hoist must carry its CONDITION along on the
// next one.
//
// cloneOpsInBlock decides what to bring by a "hoist" label, and the label comes
// from the backward slice of the transfers being moved. That slice is built out
// of SSA operands, and at this stage the operands that matter are async tokens
// -- so a region op that yields a token gets labelled, while a pure op feeding
// its condition does not: it produces no token, so no dependence edge reaches
// it. cloneOpsInBlock then skips the pure op outright (it is neither labelled
// nor async), the switch clone resolves its condition through lookupOrDefault,
// and the clone at the outer scope keeps pointing at the def at the inner one.
//
// Nothing complains at the point of damage. The pass finishes and the verifier
// reports "operand #0 does not dominate this use" against an op several
// hoists away from the one that broke it.
//
// The shape below is the two-step version, which is the only one that reaches
// it. Step one hoists @a out of herd A, rebuilding A's arm at segment scope --
// that rebuild is what creates the `arith.index_cast` + `scf.index_switch`
// pair. @a is L2-side there, so step two does not treat it as a target; it
// arrives in the slice only through the token @b waits on. Were the switch a
// PARENT of a target, the parent walk would pull its condition in and the bug
// would not fire.

// CHECK-LABEL: func.func @pure_guard_def
// The switch rebuilt at segment scope is cloned again to launch scope, and its
// condition is cloned with it. Without that, the launch-scope switch reads the
// segment-scope index_cast and the module no longer verifies.
// CHECK: air.launch
// CHECK: arith.index_cast
// CHECK: scf.index_switch

air.channel @a [1]
air.channel @b [1]
func.func @pure_guard_def(%arg0: memref<64xbf16>, %rt: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lrt=%rt) : memref<64xbf16>, i32 {
    air.segment @seg args(%sa=%la, %srt=%lrt) : memref<64xbf16>, i32 {
      %c1_s = arith.constant 1 : index
      %l2 = memref.alloc() : memref<64xbf16, 1>
      // Herd A: an L2-side transfer under an arm keyed on a runtime parameter.
      // Hoisting it rebuilds the arm at segment scope.
      %c2_s = arith.constant 2 : index
      air.herd @ha tile (%ax, %ay) in (%asx=%c2_s, %asy=%c2_s) args(%ad=%l2, %art=%srt) : memref<64xbf16, 1>, i32 {
        %c0_a = arith.constant 0 : index
        %c1_a = arith.constant 1 : index
        %c64_a = arith.constant 64 : index
        %ia = arith.index_cast %art : i32 to index
        scf.index_switch %ia
        case 0 {
          scf.yield
        }
        default {
          // Exactly one tile transfers, so the hoist places the rebuilt arm at
          // segment TOP LEVEL rather than inside an scf.parallel. That is what
          // leaves the condition exposed on the next hoist.
          %isx = arith.cmpi eq, %ax, %c0_a : index
          scf.if %isx {
            %isy = arith.cmpi eq, %ay, %c0_a : index
            scf.if %isy {
              %la1 = memref.alloc() : memref<64xbf16, 2>
              air.dma_memcpy_nd (%la1[] [] [], %ad[%c0_a] [%c64_a] [%c1_a]) {id = 1 : i32, channel = @a, channel_indices = array<i64: 0>} : (memref<64xbf16, 2>, memref<64xbf16, 1>)
              memref.dealloc %la1 : memref<64xbf16, 2>
            }
          }
          scf.yield
        }
      }
      // Herd B: an L3-side transfer, ordered after A through %l2. Its hoist to
      // launch scope is the one that pulls A's rebuilt switch along.
      air.herd @hb tile (%bx, %by) in (%bsx=%c1_s, %bsy=%c1_s) args(%bd=%l2, %bs=%sa) : memref<64xbf16, 1>, memref<64xbf16> {
        %c0_b = arith.constant 0 : index
        %c1_b = arith.constant 1 : index
        %c64_b = arith.constant 64 : index
        %lb1 = memref.alloc() : memref<64xbf16, 2>
        air.dma_memcpy_nd (%lb1[] [] [], %bs[%c0_b] [%c64_b] [%c1_b]) {id = 2 : i32, channel = @b, channel_indices = array<i64: 0>} : (memref<64xbf16, 2>, memref<64xbf16>)
        air.dma_memcpy_nd (%bd[%c0_b] [%c64_b] [%c1_b], %lb1[] [] []) {id = 3 : i32} : (memref<64xbf16, 1>, memref<64xbf16, 2>)
        memref.dealloc %lb1 : memref<64xbf16, 2>
      }
      memref.dealloc %l2 : memref<64xbf16, 1>
    }
  }
  return
}
