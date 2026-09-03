//===- memtile_per_fill_refeed.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-refeed -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s
// RUN: air-opt %s -air-annotate-refeed -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// PER-FILL air.refeed_count on ONE memtile buffer.
//
// A relay slab is filled once per phase and re-broadcast that phase's number of
// times, and the counts differ per phase. The count is therefore a property of
// the FILL, not of the buffer. Before this was expressible, each count needed
// its own buffer -- hence its own DMA chain, hence, once a tile's channels and
// packet ids ran out, its own memtile.
//
// The hardware has no such restriction, and neither template below does: ONE
// slab, one write lock, one read lock, two fill BDs with different counts and a
// count-free self-looping drain. The write lock inits to the LARGEST count, so
// a fill can only start once the drain has returned every token the previous
// fill enabled -- fill A takes 12 and the drain returns 12, fill B takes 38 and
// the drain returns 38.
//
// Both lock templates are pinned because the two disagreed here: the v2 daisy
// chain would order the fills as if they were disjoint sub-region writers, so
// fill B would acquire the tokens fill A released, which cannot balance when
// the counts differ (the second round deadlocks). Per-fill counts mark the
// fills as independent whole-buffer overwrites and opt the buffer out of the
// chain -- see air::attrs::RefeedPerFill.

// CHECK: aie.device
// CHECK: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// The write lock is primed to the LARGEST fill count, not to either one alone
// and not to a slot count.
// CHECK: %[[WR:.*]] = aie.lock(%[[MT]], {{[0-9]+}}) {init = 38 : i32}
// CHECK: %[[RD:.*]] = aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}

// Exactly ONE buffer instance: a re-broadcast has no ping-pong twin (the pong
// slot would never be filled). It carries the max count and the per-fill mark.
// CHECK: %[[BUF:.*]] = aie.buffer(%[[MT]]) {air.refeed_count = 38 : i32, air.refeed_per_fill
// CHECK-NOT: aie.buffer(%[[MT]])

// CHECK: aie.memtile_dma(%[[MT]])

// The drain moves ONE token per fire and self-loops; its rate is set by what
// the fill released, not by the number of fills.
// CHECK: aie.dma_start(MM2S, 0, ^[[DRAIN:.*]], ^{{.*}})
// CHECK: ^[[DRAIN]]:
// CHECK: aie.use_lock(%[[RD]], AcquireGreaterEqual, %c1_i32)
// CHECK: aie.dma_bd(%[[BUF]] :
// CHECK: aie.use_lock(%[[WR]], Release, %c1_i32)
// CHECK: aie.next_bd ^[[DRAIN]]

// Fill A: its own count, against the SAME buffer and the SAME lock pair.
// CHECK: aie.dma_start(S2MM, 0, ^[[FILLA:.*]], ^{{.*}})
// CHECK: ^[[FILLA]]:
// CHECK: aie.use_lock(%[[WR]], AcquireGreaterEqual, %c12_i32)
// CHECK: aie.dma_bd(%[[BUF]] :
// CHECK: aie.use_lock(%[[RD]], Release, %c12_i32)
// CHECK: aie.next_bd ^[[FILLA]]

// Fill B: a DIFFERENT count, same buffer, same locks.
// CHECK: aie.dma_start(S2MM, 1, ^[[FILLB:.*]], ^{{.*}})
// CHECK: ^[[FILLB]]:
// CHECK: aie.use_lock(%[[WR]], AcquireGreaterEqual, %c38_i32)
// CHECK: aie.dma_bd(%[[BUF]] :
// CHECK: aie.use_lock(%[[RD]], Release, %c38_i32)
// CHECK: aie.next_bd ^[[FILLB]]

air.channel @fillA [1, 1]
air.channel @fillB [1, 1]
air.channel @drain [1, 1]
func.func @memtile_per_fill_refeed() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      // One resident slab. No count on the ALLOC: that is the single-count
      // encoding this test exists to generalise.
      %t, %l2 = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      // Two fills of the WHOLE slab, each re-broadcast its own number of
      // times. As the FRONT END writes it: a fill, then an N-trip loop around the
      // re-send. air-annotate-refeed folds each loop and lands its count on the
      // get that filled the slab for it.
      %c0_i = arith.constant 0 : index
      %c12_i = arith.constant 12 : index
      %c38_i = arith.constant 38 : index
      air.channel.get @fillA[] (%l2[] [] []) : (memref<32xbf16, 1>)
      scf.for %ra = %c0_i to %c12_i step %c1_0 {
        air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      }
      air.channel.get @fillB[] (%l2[] [] []) : (memref<32xbf16, 1>)
      scf.for %rb = %c0_i to %c38_i step %c1_0 {
        air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      }
      %dd = air.execute {memref.dealloc %l2 : memref<32xbf16, 1>}
      air.herd @hA tile (%txa, %tya) in (%sxa=%c1_0, %sya=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @fillA[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %da = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
      air.herd @hB tile (%txb, %tyb) in (%sxb=%c1_0, %syb=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @fillB[] (%l1[] [] []) : (memref<32xbf16, 2>)
        %db = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
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
