//===- tile_s2mm_chain_arrival_order.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics -split-input-file

// TileDMAAllocator::verifyS2MMChains.
//
// A compute-tile S2MM BD chain is walked STRICTLY IN ORDER: a packet header
// routes a transfer to the tile, it does not select which BD receives it. So a
// chain shared by several flows stays correct only while every feasible
// control path through the consumer presents the SAME sequence of BDs. If one
// path delivers a different number (or order) of transfers than the ring was
// built for, the BD pointer slips further every dispatch until a transfer
// meets a BD belonging to another flow and the receiver deadlocks.
//
// The pass does not move anything -- it reports the chains that cannot stay in
// step. Only chains hosting two or more distinct flows can mis-deliver ACROSS
// flows, so single-flow chains are out of scope.

// -----

// Symmetric arms: both arms consume @sA then @sB, onto the same buffers. The
// two arms fold onto one 2-BD ring that matches either path. This is the shape
// the shipped llama-3.2-1B q4nx decode rms tile produces (@rmsX/@rmsW/@rmsW2
// under an scf.index_switch) and it must stay silent.

air.channel @sA [1] {channel_type = "npu_dma_packet"}
air.channel @sB [1] {channel_type = "npu_dma_packet"}
func.func @symmetric_arms(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @sA[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @sB[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %a = memref.alloc() : memref<8xbf16, 2>
        %b = memref.alloc() : memref<8xbf16, 2>
        scf.index_switch %tx
        case 0 {
          air.channel.get @sA[] (%a[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
          air.channel.get @sB[] (%b[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        default {
          air.channel.get @sA[] (%a[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
          air.channel.get @sB[] (%b[] [] []) {id = 4 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        memref.dealloc %a : memref<8xbf16, 2>
        memref.dealloc %b : memref<8xbf16, 2>
      }
    }
  }
  return
}

// -----

// Asymmetric arms: the decode arm reads the sublayer buffer twice, the lm-head
// arm once, so the two arms present words of different length. This is the
// 4-norm (Gemma-style) decode shape, and it is what deadlocks on the second
// dispatch.

air.channel @aX [1] {channel_type = "npu_dma_packet"}
air.channel @aSub [1] {channel_type = "npu_dma_packet"}
func.func @asymmetric_arms(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @aX[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @aSub[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %x = memref.alloc() : memref<8xbf16, 2>
        %sub = memref.alloc() : memref<8xbf16, 2>
        scf.index_switch %tx
        case 0 {
          air.channel.get @aX[] (%x[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
          air.channel.get @aSub[] (%sub[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
          air.channel.get @aSub[] (%sub[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        default {
          air.channel.get @aX[] (%x[] [] []) {id = 4 : i32} : (memref<8xbf16, 2>)
          air.channel.get @aSub[] (%sub[] [] []) {id = 5 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        memref.dealloc %x : memref<8xbf16, 2>
        memref.dealloc %sub : memref<8xbf16, 2>
      }
    }
  }
  return
}

// -----

// A hole: only one arm consumes @hW. The other arm leaves the chain a transfer
// short, which is precisely what the hand-written dummy get in the llama q4nx
// fused decode (`_uni_voc` feeding a dummy @rmsW2) exists to prevent.

air.channel @hX [1] {channel_type = "npu_dma_packet"}
air.channel @hW [1] {channel_type = "npu_dma_packet"}
func.func @arm_hole(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @hX[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @hW[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %x = memref.alloc() : memref<8xbf16, 2>
        %w = memref.alloc() : memref<8xbf16, 2>
        scf.index_switch %tx
        case 0 {
          air.channel.get @hX[] (%x[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
          air.channel.get @hW[] (%w[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        default {
          air.channel.get @hX[] (%x[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        memref.dealloc %x : memref<8xbf16, 2>
        memref.dealloc %w : memref<8xbf16, 2>
      }
    }
  }
  return
}

// -----

// A single-flow chain cannot mis-deliver across flows, so an asymmetric single
// -flow chain is out of scope and must stay silent. Left un-warned on purpose:
// the shipped llama q4nx decode has exactly this shape on its @outY chain.

air.channel @oY [1] {channel_type = "npu_dma_packet"}
func.func @single_flow_asymmetric(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @oY[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %y = memref.alloc() : memref<8xbf16, 2>
        scf.index_switch %tx
        case 0 {
          air.channel.get @oY[] (%y[0] [4] [1]) {id = 1 : i32} : (memref<8xbf16, 2>)
          air.channel.get @oY[] (%y[4] [4] [1]) {id = 2 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        default {
          air.channel.get @oY[] (%y[0] [4] [1]) {id = 3 : i32} : (memref<8xbf16, 2>)
          scf.yield
        }
        memref.dealloc %y : memref<8xbf16, 2>
      }
    }
  }
  return
}

// -----

// Straight-line multi-flow chain with unequal multiplicity, all transfers
// mapping to interchangeable BDs. There is no branch, so every dispatch
// delivers the same word and the chain is in step. #1771's multiplicity
// heuristic split this; it must not be reported.

air.channel @eA [1] {channel_type = "npu_dma_packet"}
air.channel @eB [1] {channel_type = "npu_dma_packet"}
func.func @unequal_multiplicity_straight_line(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @eA[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @eB[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.channel.put @eB[] (%e[] [] []) {id = 3 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %a = memref.alloc() : memref<8xbf16, 2>
        %b = memref.alloc() : memref<8xbf16, 2>
        air.channel.get @eA[] (%a[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.get @eB[] (%b[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
        air.channel.get @eB[] (%b[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        memref.dealloc %a : memref<8xbf16, 2>
        memref.dealloc %b : memref<8xbf16, 2>
      }
    }
  }
  return
}
