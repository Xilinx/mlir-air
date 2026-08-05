//===- tile_aperiodic_packet_chain_split.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// TileDMAAllocator::rebalanceAperiodicPacketChains.
//
// A compute-tile S2MM BD chain is walked STRICTLY IN ORDER: a packet header
// routes a transfer to the tile, it does not select which BD receives it. So a
// chain shared by several packet flows is only correct while a transfer can
// never meet a BD belonging to a different flow -- i.e. while the chain is
// HOMOGENEOUS (one flow) or PERIODIC (flows interleave, and the sequence folds
// into a circular ring whose period is the per-dispatch arrival pattern).
//
// Mixing flows that are consumed a DIFFERENT number of times per dispatch makes
// the sequence aperiodic: getRepeatCounts finds no repeating pattern, one BD is
// emitted per transfer, and the ring length no longer matches arrival, so the
// BD pointer drifts a little further every dispatch until a transfer meets a
// BD expecting another flow and the receiver deadlocks. The allocator splits
// such a chain across a spare S2MM channel of the same tile.

// -----

// Aperiodic: @cA arrives once per dispatch, @cB twice -> [A, B, B]. The packet
// collapse would put all three on S2MM 0; the rebalance peels one flow onto the
// tile's spare S2MM channel so both chains are homogeneous.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK-DAG: aie.dma_start(S2MM, 0
// CHECK-DAG: aie.dma_start(S2MM, 1
// The flows must follow the BDs onto both channels, otherwise the switchbox
// routes to a channel that has no BD for it.
// CHECK-DAG: aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK-DAG: aie.packet_dest<%{{.*}}, DMA : 1>

air.channel @cA [1] {channel_type = "npu_dma_packet"}
air.channel @cB [1] {channel_type = "npu_dma_packet"}
func.func @aperiodic_split(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @cA[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @cB[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.channel.put @cB[] (%e[] [] []) {id = 3 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %t0, %a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        %t1, %b = air.execute -> (memref<8xbf16, 2>) {
          %bb = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %bb : memref<8xbf16, 2>
        }
        air.channel.get @cA[] (%a[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.get @cB[] (%b[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
        air.channel.get @cB[] (%b[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// Periodic: @cA and @cB alternate -> [A, B, A, B] folds into a 2-BD circular
// ring whose period matches arrival, so the shared channel is already correct.
// This is the shape a working 2-norm decode produces; it must NOT be split
// (an extra channel here would be a regression, not a fix).

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NOT: aie.dma_start(S2MM, 1

air.channel @pA [1] {channel_type = "npu_dma_packet"}
air.channel @pB [1] {channel_type = "npu_dma_packet"}
func.func @periodic_no_split(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @pA[] (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @pB[] (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.channel.put @pA[] (%e[] [] []) {id = 3 : i32} : (memref<8xbf16>)
    air.channel.put @pB[] (%e[] [] []) {id = 4 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %t0, %a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        %t1, %b = air.execute -> (memref<8xbf16, 2>) {
          %bb = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %bb : memref<8xbf16, 2>
        }
        air.channel.get @pA[] (%a[] [] []) {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.get @pB[] (%b[] [] []) {id = 2 : i32} : (memref<8xbf16, 2>)
        air.channel.get @pA[] (%a[] [] []) {id = 3 : i32} : (memref<8xbf16, 2>)
        air.channel.get @pB[] (%b[] [] []) {id = 4 : i32} : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// The shape a 4-norm (Gemma-style) decode rms tile produces. The herd body has
// two mutually-exclusive arms (decode / lm-head), so each of the three
// norm-weight flows is read twice while the sublayer flow -- the o-proj result,
// then the down-proj result, through one buffer -- is read three times:
//   [X, W, W2, sub, sub, X, W, W2, sub]
// Packet collapse puts all four flows on S2MM 0, where that sequence has no
// repeating period. The weights alone DO repeat ([X, W, W2] twice), so they
// stay together on one circular chain and the odd-multiplicity sublayer is
// moved to the tile's other S2MM channel.

// CHECK-LABEL: aie.device
// CHECK: aie.mem
// Weights: one chain carrying all three distinct pkt_ids.
// CHECK: aie.dma_start(S2MM, 0
// CHECK-DAG: pkt_id = 0
// CHECK-DAG: pkt_id = 1
// CHECK-DAG: pkt_id = 2
// Sublayer: isolated on the spare channel.
// CHECK: aie.dma_start(S2MM, 1
// CHECK: aie.dma_bd(%{{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>
// CHECK-NOT: aie.dma_start(S2MM, 0

air.channel @rX [1] {channel_type = "npu_dma_packet"}
air.channel @rW [1] {channel_type = "npu_dma_packet"}
air.channel @rW2 [1] {channel_type = "npu_dma_packet"}
air.channel @rSub [1] {channel_type = "npu_dma_packet"}
func.func @rms_four_norm_shape(%ext: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<8xbf16> {
    air.channel.put @rX[]   (%e[] [] []) {id = 1 : i32} : (memref<8xbf16>)
    air.channel.put @rW[]   (%e[] [] []) {id = 2 : i32} : (memref<8xbf16>)
    air.channel.put @rW2[]  (%e[] [] []) {id = 3 : i32} : (memref<8xbf16>)
    air.channel.put @rSub[] (%e[] [] []) {id = 4 : i32} : (memref<8xbf16>)
    air.channel.put @rSub[] (%e[] [] []) {id = 5 : i32} : (memref<8xbf16>)
    air.channel.put @rX[]   (%e[] [] []) {id = 6 : i32} : (memref<8xbf16>)
    air.channel.put @rW[]   (%e[] [] []) {id = 7 : i32} : (memref<8xbf16>)
    air.channel.put @rW2[]  (%e[] [] []) {id = 8 : i32} : (memref<8xbf16>)
    air.channel.put @rSub[] (%e[] [] []) {id = 9 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @rms tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %t0, %x = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        %t1, %w = air.execute -> (memref<8xbf16, 2>) {
          %bb = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %bb : memref<8xbf16, 2>
        }
        %t2, %w2 = air.execute -> (memref<8xbf16, 2>) {
          %cc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %cc : memref<8xbf16, 2>
        }
        %t3, %sub = air.execute -> (memref<8xbf16, 2>) {
          %dd = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %dd : memref<8xbf16, 2>
        }
        // decode arm: three norm weights + the sublayer twice.
        air.channel.get @rX[]   (%x[] [] [])   {id = 1 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rW[]   (%w[] [] [])   {id = 2 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rW2[]  (%w2[] [] [])  {id = 3 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rSub[] (%sub[] [] []) {id = 4 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rSub[] (%sub[] [] []) {id = 5 : i32} : (memref<8xbf16, 2>)
        // lm-head arm: the same weights again, sublayer once.
        air.channel.get @rX[]   (%x[] [] [])   {id = 6 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rW[]   (%w[] [] [])   {id = 7 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rW2[]  (%w2[] [] [])  {id = 8 : i32} : (memref<8xbf16, 2>)
        air.channel.get @rSub[] (%sub[] [] []) {id = 9 : i32} : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %x : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %w : memref<8xbf16, 2>}
        %d2 = air.execute {memref.dealloc %w2 : memref<8xbf16, 2>}
        %d3 = air.execute {memref.dealloc %sub : memref<8xbf16, 2>}
      }
    }
  }
  return
}
