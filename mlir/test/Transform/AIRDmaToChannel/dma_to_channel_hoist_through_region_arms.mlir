//===- dma_to_channel_hoist_through_region_arms.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Hoisting an external channel op out of a region-holding op (scf.index_switch
// arm, scf.if branch) has to rebuild that op AND give the arm a token that
// names only values defined in the arm itself.
//
// The bug these pin: the arm's token was built from cloneOpsInBlock's return
// value, which lists what was cloned at EVERY depth -- that is how the caller
// finds channel ops buried in a hoisted loop body. So as soon as an arm held a
// loop, the arm's wait_all named tokens defined INSIDE that loop's region, and
// the module failed to verify with "operand #1 does not dominate this use".
//
// The trigger is one arm containing a loop, not any particular depth of
// nesting: hoisting a herd out of a switch arm leaves the herd's scf.parallel
// in that arm, so a SINGLE switch is enough. The first case below is that
// minimal shape. It was previously misdiagnosed as requiring two levels.

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// A herd inside a switch arm. Hoisting the herd leaves an scf.parallel in the
// arm, so the arm's yield token must not reach into it. One switch only.
// CHECK-LABEL: func.func @one_switch
// CHECK: air.launch
// CHECK: scf.index_switch
// CHECK: scf.parallel
// CHECK: air.channel.put{{.*}}@c
air.channel @c []
func.func @one_switch(%arg0: memref<64x64xi32>, %sw: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %s=%sw) : memref<64x64xi32>, index {
    air.segment args(%sa=%la, %ss=%s) : memref<64x64xi32>, index {
      %c1_0 = arith.constant 1 : index
      scf.index_switch %ss
      case 0 {
        scf.yield
      }
      default {
        air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c64 = arith.constant 64 : index
          %cst1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @c} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          memref.dealloc %alloc : memref<32x32xi32, 2>
        }
        scf.yield
      }
    }
  }
  return
}

// -----

// Both levels switched: the DMA sits in a switch arm inside a herd, and that
// herd sits in a switch arm of its own inside the segment. Both switches must
// survive, and the put must stay on the arm it was written for rather than
// being flattened onto every arm. This is the shape a per-layer-type feed takes
// in the fused_decode builder.
// CHECK-LABEL: func.func @two_switches
// CHECK: air.launch
// CHECK: scf.index_switch
// CHECK: scf.parallel
// CHECK: scf.index_switch
// CHECK: air.channel.put{{.*}}@c
// CHECK: air.herd
// CHECK: air.channel.get{{.*}}@c
air.channel @c []
func.func @two_switches(%arg0: memref<64x64xi32>, %sw: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %s=%sw) : memref<64x64xi32>, index {
    air.segment args(%sa=%la, %ss=%s) : memref<64x64xi32>, index {
      %c1_0 = arith.constant 1 : index
      scf.index_switch %ss
      case 0 {
        scf.yield
      }
      default {
        air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa, %sv=%ss) : memref<64x64xi32>, index {
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c64 = arith.constant 64 : index
          %cst1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          scf.index_switch %sv
          case 0 {
            scf.yield
          }
          default {
            air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @c} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            scf.yield
          }
          memref.dealloc %alloc : memref<32x32xi32, 2>
        }
        scf.yield
      }
    }
  }
  return
}

// -----

// The same defect reached through scf.if: cloneScfIfUsingRemap builds the
// replacement token for a result-carrying scf.if the same way, so a herd inside
// an scf.if branch hit it too. air-dependency gives the scf.if a token result,
// which is what selects that path.
// CHECK-LABEL: func.func @scf_if_branch
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@c
air.channel @c []
func.func @scf_if_branch(%arg0: memref<64x64xi32>, %cnd: i1) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %cc=%cnd) : memref<64x64xi32>, i1 {
    air.segment args(%sa=%la, %sc=%cc) : memref<64x64xi32>, i1 {
      %c1_0 = arith.constant 1 : index
      scf.if %sc {
        air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c64 = arith.constant 64 : index
          %cst1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<32x32xi32, 2>
          air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @c} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          memref.dealloc %alloc : memref<32x32xi32, 2>
        }
      }
    }
  }
  return
}
