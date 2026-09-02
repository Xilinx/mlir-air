//===- dma_to_channel_hoist_nested_arm_anchor.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// An anchor endpoint one arm DEEPER than the transfer is still in the right
// arm. Land ahead of the structure that holds them, not inside one of them.
//
// An anchor is matched by the PATH of switch arms it sits under, because a bare
// arm index treats two different switches' arm 0 as the same place. But exact
// path equality is too strong the other way: a feed that is split per column
// group has its endpoints inside a NESTED switch, at [0, 0..3], while the
// transfer naming it sits plainly in [0]. No path is equal, the match falls
// through to "any arm", and the first endpoint anywhere wins -- on llama-3.1-8b
// that is the endpoint in the OTHER outer arm, and since two more channels
// anchor onto this one in turn, three feeds move past the weight stream.
//
// Ranking exact > inside > any fixes the match. Landing ON an inside match
// would then put the transfer in ONE of the nested arms and starve the rest --
// the failure that withdrew RMSW_DMA on qwen3_8b once already -- so an inside
// match climbs back out to the switch that encloses them. Before that switch
// is ahead of every nested arm's first endpoint, which is what "before the
// anchor" means when the anchor is spread across arms.

// CHECK-LABEL: func.func @nested_arm_anchor
// The derived put lands in the outer default arm, AHEAD of the nested switch
// that holds @w's endpoints -- not inside either of its arms, and not over in
// the other outer arm with the decoy.
// CHECK: air.launch
// CHECK: scf.index_switch
// CHECK: air.channel.put{{.*}}@t
// CHECK: scf.index_switch
// CHECK: air.channel.put{{.*}}@w
// CHECK: air.channel.put{{.*}}@w

air.channel @w [1]
air.channel @t [1]
func.func @nested_arm_anchor(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %sel: index, %sub: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lb=%arg1, %s=%sel, %u=%sub) : memref<64xbf16>, memref<64xbf16>, index, index {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.index_switch %s
    case 0 {
      // The decoy: @w in the OTHER outer arm. Its path is [1], which is not a
      // prefix of anything, so it must never be chosen over the nested ones.
      air.channel.put @w[%c0] (%lb[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
      scf.yield
    }
    default {
      // @w split per group: every endpoint in this arm is one level deeper.
      scf.index_switch %u
      case 0 {
        air.channel.put @w[%c0] (%lb[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
        scf.yield
      }
      default {
        air.channel.put @w[%c0] (%lb[%c0] [%c64] [%c1_l]) : (memref<64xbf16>)
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
          scf.yield
        }
        default {
          // Sits plainly in the outer default arm: path [0], one shallower than
          // any @w endpoint in that arm.
          %l1 = memref.alloc() : memref<64xbf16, 2>
          air.dma_memcpy_nd (%l1[] [] [], %b[%c0_h] [%c64_h] [%c1_h]) {id = 1 : i32, channel = @t, channel_indices = array<i64: 0>, hoist_before = @w} : (memref<64xbf16, 2>, memref<64xbf16>)
          memref.dealloc %l1 : memref<64xbf16, 2>
          scf.yield
        }
      }
    }
  }
  return
}
