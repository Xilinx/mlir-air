//===- tile_producer_partition.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" -split-input-file | FileCheck %s

// TileDMAAllocator::spreadCollapsedPacketChannels, producer rule.
//
// A compute-tile DMA channel is ONE BD ring walked strictly in order, and a
// packet header steers a transfer to the tile without choosing which BD takes
// it. When the flows on a ring come from producers that are not synchronised
// with each other, what keeps it correct is that the ring is a ROUND-ROBIN:
// every flow owning exactly one BD of the repeating unit, so a flow's k-th
// transfer always meets that same BD however the arrivals interleave.
//
// Let one flow own two BDs of a cycle and that stops holding -- the ring
// position is no longer a function of how many rounds have happened, and an
// arrival can land on a BD bound to another flow's buffer and lock. The repair
// is a partition by producer, after which each ring is ordered by its own
// producer again.
//
// Both cases below need more than one column, because the two L2 buckets have
// to land on DIFFERENT memtiles to be different producers at all. On
// npu1_1col they share one memtile, become one producer, and correctly stay
// collapsed -- which is the same rule giving the other answer.

// -----

// Two producers, NOT a round-robin: the a-leg owns two BDs of the cycle. This
// is the shape the qwen2.5-3b rms core produces; left collapsed it builds,
// ROUTES, passes diagnoseBDChain, and deadlocks on device.

// CHECK-LABEL: aie.device
// CHECK:      aie.mem
// CHECK:        aie.dma_start(S2MM, 0
// CHECK:        aie.dma_bd(%{{.*}} : memref<8x8xi32, 2>
// CHECK:        aie.dma_bd(%{{.*}} : memref<8x8xi32, 2>
// CHECK:        aie.dma_start(S2MM, 1
// CHECK:        aie.dma_bd(%{{.*}} : memref<4x16xi32, 2>

  // Two L3->L2 packet channels feeding col 0 via memtiles (operand
  // classes A and B). Two buckets at col 0 trigger the saturation
  // fallback in L2MemrefToMemTileMap, leaving memtiles col-unhinted.
  air.channel @pkt_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkt_b [1, 1] {channel_type = "npu_dma_packet"}
  // One L3->L1 direct packet channel feeding col 0.
  // L2->L1 legs anchoring the memtile-routed flows to col 0 cores.
  air.channel @a_l2_to_core [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @b_l2_to_core [1, 1] {channel_type = "npu_dma_packet"}

  func.func @func_shared_col(%a: memref<64xi32>, %b: memref<64xi32>,
                              %c: memref<64xi32>) {
    air.channel.put @pkt_a[] (%a[] [] []) {id = 1 : i32} : (memref<64xi32>)
    air.channel.put @pkt_b[] (%b[] [] []) {id = 2 : i32} : (memref<64xi32>)
    air.segment @seg attributes {x_loc = 0 : i64, x_size = 4 : i64,
                                  y_loc = 2 : i64, y_size = 1 : i64} {
      %c1 = arith.constant 1 : index
      // Two distinct operand-class shapes (8 vs 16 elements) so the two
      // buckets stay disjoint in L2MemrefToMemTileMap.
      %l2_a = memref.alloc() : memref<8x8xi32, 1>
      %l2_b = memref.alloc() : memref<4x16xi32, 1>
      air.channel.get @pkt_a[] (%l2_a[] [] []) {id = 4 : i32} : (memref<8x8xi32, 1>)
      air.channel.get @pkt_b[] (%l2_b[] [] []) {id = 5 : i32} : (memref<4x16xi32, 1>)
      air.channel.put @a_l2_to_core[] (%l2_a[] [] []) {id = 6 : i32} : (memref<8x8xi32, 1>)
      air.channel.put @a_l2_to_core[] (%l2_a[] [] []) {id = 12 : i32} : (memref<8x8xi32, 1>)
      air.channel.put @b_l2_to_core[] (%l2_b[] [] []) {id = 7 : i32} : (memref<4x16xi32, 1>)
      air.herd @h tile(%tx, %ty) in (%sx = %c1, %sy = %c1)
          attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %l1_a = memref.alloc() : memref<8x8xi32, 2>
        %l1_b = memref.alloc() : memref<4x16xi32, 2>
        %l1_a2 = memref.alloc() : memref<8x8xi32, 2>
        air.channel.get @a_l2_to_core[%tx, %ty] (%l1_a[] [] []) {id = 8 : i32} : (memref<8x8xi32, 2>)
        air.channel.get @a_l2_to_core[%tx, %ty] (%l1_a2[] [] []) {id = 11 : i32} : (memref<8x8xi32, 2>)
        memref.dealloc %l1_a2 : memref<8x8xi32, 2>
        air.channel.get @b_l2_to_core[%tx, %ty] (%l1_b[] [] []) {id = 9 : i32} : (memref<4x16xi32, 2>)
        memref.dealloc %l1_a : memref<8x8xi32, 2>
        memref.dealloc %l1_b : memref<4x16xi32, 2>
      }
      memref.dealloc %l2_a : memref<8x8xi32, 1>
      memref.dealloc %l2_b : memref<4x16xi32, 1>
    }
    return
  }

// -----

// The same two producers with one BD each: a round-robin, safe however the two
// interleave, and left collapsed. Every AIR matmul is this shape -- two
// memtiles feeding a core one A tile and one B tile per iteration -- so
// splitting on producer count alone would spend both S2MM channels of every
// compute tile in the project.

// CHECK-LABEL: aie.device
// CHECK:      aie.mem
// CHECK:        aie.dma_start(S2MM, 0
// CHECK-NOT:    aie.dma_start(S2MM, 1

  // Two L3->L2 packet channels feeding col 0 via memtiles (operand
  // classes A and B). Two buckets at col 0 trigger the saturation
  // fallback in L2MemrefToMemTileMap, leaving memtiles col-unhinted.
  air.channel @pkt_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @pkt_b [1, 1] {channel_type = "npu_dma_packet"}
  // One L3->L1 direct packet channel feeding col 0.
  // L2->L1 legs anchoring the memtile-routed flows to col 0 cores.
  air.channel @a_l2_to_core [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @b_l2_to_core [1, 1] {channel_type = "npu_dma_packet"}

  func.func @func_shared_col(%a: memref<64xi32>, %b: memref<64xi32>,
                              %c: memref<64xi32>) {
    air.channel.put @pkt_a[] (%a[] [] []) {id = 1 : i32} : (memref<64xi32>)
    air.channel.put @pkt_b[] (%b[] [] []) {id = 2 : i32} : (memref<64xi32>)
    air.segment @seg attributes {x_loc = 0 : i64, x_size = 4 : i64,
                                  y_loc = 2 : i64, y_size = 1 : i64} {
      %c1 = arith.constant 1 : index
      // Two distinct operand-class shapes (8 vs 16 elements) so the two
      // buckets stay disjoint in L2MemrefToMemTileMap.
      %l2_a = memref.alloc() : memref<8x8xi32, 1>
      %l2_b = memref.alloc() : memref<4x16xi32, 1>
      air.channel.get @pkt_a[] (%l2_a[] [] []) {id = 4 : i32} : (memref<8x8xi32, 1>)
      air.channel.get @pkt_b[] (%l2_b[] [] []) {id = 5 : i32} : (memref<4x16xi32, 1>)
      air.channel.put @a_l2_to_core[] (%l2_a[] [] []) {id = 6 : i32} : (memref<8x8xi32, 1>)
      air.channel.put @b_l2_to_core[] (%l2_b[] [] []) {id = 7 : i32} : (memref<4x16xi32, 1>)
      air.herd @h tile(%tx, %ty) in (%sx = %c1, %sy = %c1)
          attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %l1_a = memref.alloc() : memref<8x8xi32, 2>
        %l1_b = memref.alloc() : memref<4x16xi32, 2>
        air.channel.get @a_l2_to_core[%tx, %ty] (%l1_a[] [] []) {id = 8 : i32} : (memref<8x8xi32, 2>)
        air.channel.get @b_l2_to_core[%tx, %ty] (%l1_b[] [] []) {id = 9 : i32} : (memref<4x16xi32, 2>)
        memref.dealloc %l1_a : memref<8x8xi32, 2>
        memref.dealloc %l1_b : memref<4x16xi32, 2>
      }
      memref.dealloc %l2_a : memref<8x8xi32, 1>
      memref.dealloc %l2_b : memref<4x16xi32, 1>
    }
    return
  }
