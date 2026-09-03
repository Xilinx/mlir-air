//===- memtile_per_fill_refeed_pingpong.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-refeed -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// TWO SLABS PING-PONGED on one channel: fill A, drain A x12, fill B, drain B x38.
//
// memtile_per_fill_refeed_onechan.mlir is the one-slab form, where a fill has to
// acquire the PREVIOUS fill's count because both fills share a lock pair. Here
// they do not: each slab has its own pair, so "how many re-sends have finished"
// is a question about that slab alone, and the previous fill of THIS slab is the
// fill itself. Taking the immediate chain predecessor instead would make slab A
// acquire 38 against a write lock that inits to 12 -- a lock that can never be
// satisfied, i.e. a hang with nothing wrong at compile time.
//
// WHY A RELAY WANTS TWO SLABS. With one, fill N+1's BD is not armed until fill
// N's re-sends have all gone, so the producing core's next transfer stalls -- and
// a compute tile's channel.put lowers to `call; acquire(avail); release(ready)`,
// with the acquire AFTER the write, so the core has already overwritten the
// staging buffer before it blocks. The stalled transfer then carries the wrong
// row. Measured on qwen3-4b's LM head: token 0's logits came back a blend of
// rows 0 and 1 (corr 0.845 to each, against 0.992/0.681 when correct) while rows
// 1..7 stayed bit-identical. A second slab is always armed, so nothing stalls.

// CHECK: aie.device
// CHECK: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)

// One lock pair per slab, each write lock initialised to ITS OWN count.
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 12 : i32}
// CHECK-DAG: aie.lock(%[[MT]], {{[0-9]+}}) {init = 38 : i32}

// CHECK: aie.memtile_dma(%[[MT]])

// Two count-free drains, one per slab, alternating.
// CHECK: aie.dma_start(MM2S, 0, ^[[DA:.*]], ^{{.*}})
// CHECK: ^[[DA]]:
// CHECK: aie.use_lock({{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK: aie.next_bd ^[[DB:.*]]
// CHECK: ^[[DB]]:
// CHECK: aie.use_lock({{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK: aie.next_bd ^[[DA]]

// Each fill acquires and releases ITS OWN count -- 12 against 12, 38 against 38.
// CHECK: aie.dma_start(S2MM, 0, ^[[FA:.*]], ^{{.*}})
// CHECK: ^[[FA]]:
// CHECK: aie.use_lock(%[[WA:.*]], AcquireGreaterEqual, %c12_i32)
// CHECK: aie.dma_bd(%[[BA:.*]] :
// CHECK: aie.use_lock(%[[RA:.*]], Release, %c12_i32)
// CHECK: aie.next_bd ^[[FB:.*]]
// CHECK: ^[[FB]]:
// CHECK: aie.use_lock(%[[WB:.*]], AcquireGreaterEqual, %c38_i32)
// CHECK: aie.dma_bd(%[[BB:.*]] :
// CHECK: aie.use_lock(%[[RB:.*]], Release, %c38_i32)
// CHECK: aie.next_bd ^[[FA]]

air.channel @fill [1, 1]
air.channel @drain [1, 1]
func.func @memtile_per_fill_refeed_pingpong() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0_i = arith.constant 0 : index
      %c12_i = arith.constant 12 : index
      %c38_i = arith.constant 38 : index
      %ta, %sa = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      %tb, %sb = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      // Same channel and same descriptor as each other; different slab, and a
      // different count per slab.
      air.channel.get @fill[] (%sa[] [] []) : (memref<32xbf16, 1>)
      scf.for %ra = %c0_i to %c12_i step %c1_0 {
        air.channel.put @drain[] (%sa[] [] []) : (memref<32xbf16, 1>)
      }
      air.channel.get @fill[] (%sb[] [] []) : (memref<32xbf16, 1>)
      scf.for %rb = %c0_i to %c38_i step %c1_0 {
        air.channel.put @drain[] (%sb[] [] []) : (memref<32xbf16, 1>)
      }
      %dda = air.execute {memref.dealloc %sa : memref<32xbf16, 1>}
      %ddb = air.execute {memref.dealloc %sb : memref<32xbf16, 1>}
      // One producer core, one port, two sends.
      air.herd @hp tile (%txa, %tya) in (%sxa=%c1_0, %sya=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        air.channel.put @fill[] (%l1[] [] []) : (memref<32xbf16, 2>)
        air.channel.put @fill[] (%l1[] [] []) : (memref<32xbf16, 2>)
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
