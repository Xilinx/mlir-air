//===- demux_leaf_keeps_own_channel.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-refeed -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" -split-input-file | FileCheck %s

// TWO LEAVES OF ONE DEMUX FILLING ONE SLAB, each on its own S2MM chain.
//
// A channel declaration is normally one flow, so allocNewDmaChannel folds every
// endpoint of it on a tile onto that tile's allocation -- and that fold runs
// AFTER simpleDmaChannelAlloc has picked a channel, so it discards the pick.
// An indexed demux is not one flow: `packet_ids` names one routing id per
// destination, so destination i is its own flow. Folded together the two leaves
// land on ONE BD chain and alternate BDs, which is what a demux is used to
// avoid, and an air.memtile_dma_channel_min steer aimed at parting them is
// silently dropped.
//
// This is the relay shape it was found on: a producer core with one port ships
// a slab down one channel, naming leaf 0 for the counted re-feed cycle and leaf
// 1 for a single extra fill that every dispatch makes. Same buffer both leaves
// -- so the drain stays ONE count-free self-looping BD -- but the leaf-1 fill
// must not advance leaf 0's cycle.

// One buffer, so one count-free self-looping drain.
// CHECK-LABEL: aie.device(xcve2802) @seg
// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {air.refeed_count = 38 : i32, air.refeed_per_fill
// CHECK-NOT: aie.buffer(%{{.*}}) {air.refeed_count
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[DRAIN:.*]], ^{{.*}})
// CHECK: ^[[DRAIN]]:
// CHECK: aie.next_bd ^[[DRAIN]]

// LEAF 0 keeps its two-BD cycle on its own channel.
// CHECK: aie.dma_start(S2MM, 0, ^[[F1:.*]], ^{{.*}})
// CHECK: ^[[F1]]:
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c38_i32)
// CHECK: aie.use_lock(%{{.*}}, Release, %c12_i32)
// CHECK: aie.next_bd ^[[F2:.*]]
// CHECK: ^[[F2]]:
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c12_i32)
// CHECK: aie.use_lock(%{{.*}}, Release, %c38_i32)
// CHECK: aie.next_bd ^[[F1]]

// LEAF 1 gets a channel of its own, at the requested floor, and self-loops on
// its own single-transfer count.
// CHECK: aie.dma_start(S2MM, 4, ^[[F3:.*]], ^{{.*}})
// CHECK: ^[[F3]]:
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK: aie.dma_bd(%[[BUF]] :
// CHECK: aie.use_lock(%{{.*}}, Release, %c1_i32)
// CHECK: aie.next_bd ^[[F3]]

air.channel @fill [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [3 : i32, 4 : i32]}
air.channel @drain [1, 1]
func.func @demux_leaf_keeps_own_channel() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0_i = arith.constant 0 : index
      %c1_i = arith.constant 1 : index
      %c12_i = arith.constant 12 : index
      %c38_i = arith.constant 38 : index
      %t, %l2 = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      air.channel.get @fill[%c0_i, %c0_i] (%l2[] [] []) : (memref<32xbf16, 1>)
      scf.for %ra = %c0_i to %c12_i step %c1_0 {
        air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      }
      // The second leaf: same slab, one fill, one re-send. The count is spelled
      // out because a single put is not a loop for air-annotate-refeed to read,
      // and a fill with no count of its own falls back to the buffer's 38.
      air.channel.get @fill[%c0_i, %c1_i] (%l2[] [] []) {air.memtile_dma_channel_min = 3 : i32, air.refeed_count = 1 : i32} : (memref<32xbf16, 1>)
      air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      air.channel.get @fill[%c0_i, %c0_i] (%l2[] [] []) : (memref<32xbf16, 1>)
      scf.for %rb = %c0_i to %c38_i step %c1_0 {
        air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      }
      %dd = air.execute {memref.dealloc %l2 : memref<32xbf16, 1>}
      // One producer core, one port: it names the leaf per transfer with `dest`
      // rather than opening a second channel it has no port for.
      air.herd @hp tile (%txa, %tya) in (%sxa=%c1_0, %sya=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @fill[%c0_h, %c0_h] (%l1[] [] []) dest(%c0_h) : (memref<32xbf16, 2>)
        air.channel.put @fill[%c0_h, %c0_h] (%l1[] [] []) dest(%c1_h) : (memref<32xbf16, 2>)
        air.channel.put @fill[%c0_h, %c0_h] (%l1[] [] []) dest(%c0_h) : (memref<32xbf16, 2>)
        %da = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.get @drain[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
    }
  }
  return
}

// -----

// THE CASE THAT MUST NOT CHANGE: a SPATIAL broadcast. One id serves both
// destinations, so the two gets on one tile are one flow and must keep sharing
// that tile's allocation and its BD chain -- exactly what the decl-keyed reuse
// is for. Only a multi-id (indexed demux) decl is parted.

// CHECK-LABEL: aie.device(xcve2802) @bseg
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NOT: aie.dma_start(S2MM, {{[1-9]}}

air.channel @bfill [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header, packet_ids = [5 : i32]}
air.channel @bdrain [1, 1]
func.func @broadcast_leaves_still_share() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @bseg {
      %c1_0 = arith.constant 1 : index
      %c0_i = arith.constant 0 : index
      %c1_i = arith.constant 1 : index
      %t, %l2 = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      air.channel.get @bfill[%c0_i, %c0_i] (%l2[] [] []) : (memref<32xbf16, 1>)
      air.channel.put @bdrain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      air.channel.get @bfill[%c0_i, %c1_i] (%l2[] [] []) : (memref<32xbf16, 1>)
      air.channel.put @bdrain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      %dd = air.execute {memref.dealloc %l2 : memref<32xbf16, 1>}
      air.herd @hp tile (%txa, %tya) in (%sxa=%c1_0, %sya=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @bfill[%c0_h, %c0_h] (%l1[] [] []) : (memref<32xbf16, 2>)
        %da = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.get @bdrain[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
    }
  }
  return
}
