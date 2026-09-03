//===- dma_to_channel_anchor_nested_arm.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// An anchored transfer must land in ITS OWN arm, not in a same-numbered arm of
// a different scf.index_switch.
//
// The anchor resolver matches the arm rather than walk order, because
// scf.index_switch numbers its default region 0 and its cases 1..n but prints
// the cases first. Matching the INNERMOST arm index alone is not enough: two
// different switches both have a region 1, so a transfer guarded by the outer
// switch's case 0 resolves onto an endpoint sitting in a NESTED switch's case 0
// and is emitted down there. The arm it belonged to then has no producer at all
// and another arm has two -- op counts, channel counts and flow counts all stay
// the same, and the starved arm's consumer waits forever.
//
// That is what withdrew RMSW_DMA from fused_decode: on qwen3_8b_q4nx the outer
// vocab arm's @rmsW put vanished and a duplicate appeared two switches down, and
// decode hung. Six of nine models were unaffected, because a design needs
// nested guards on the anchor channel to expose it.
//
// So: @a has an endpoint in the outer case 0 and another in the nested case 0,
// and the herd-side DMA sits under case 0 of the outer switch. It belongs beside
// the first.

// CHECK-LABEL: func.func @nested
// CHECK: scf.index_switch
// CHECK-NEXT: case 0 {
// The derived put lands here, ahead of the outer arm's own @a endpoint.
// CHECK-NEXT: air.channel.put{{.*}}@t
// CHECK-NEXT: air.channel.put{{.*}}@a
// CHECK: default {
// The nested arm keeps only its own endpoint.
// CHECK: scf.index_switch
// CHECK-NEXT: case 0 {
// CHECK-NEXT: air.channel.put{{.*}}@a
// CHECK-NOT: air.channel.put{{.*}}@t

air.channel @a [1]
air.channel @t [1]
func.func @nested(%arg0: memref<64xbf16>, %sel: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %s=%sel) : memref<64xbf16>, index {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.index_switch %s
    case 0 {
      air.channel.put @a[%c0] (%la[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
      scf.yield
    }
    default {
      scf.index_switch %s
      case 0 {
        air.channel.put @a[%c0] (%la[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
        scf.yield
      }
      default {
        scf.yield
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
          %l1 = memref.alloc() : memref<64xbf16, 2>
          air.dma_memcpy_nd (%l1[] [] [], %b[%c0_h] [%c64_h] [%c1_h]) {id = 1 : i32, channel = @t, channel_indices = array<i64: 0>, hoist_before = @a} : (memref<64xbf16, 2>, memref<64xbf16>)
          memref.dealloc %l1 : memref<64xbf16, 2>
          scf.yield
        }
        default {
          scf.yield
        }
      }
    }
  }
  return
}
