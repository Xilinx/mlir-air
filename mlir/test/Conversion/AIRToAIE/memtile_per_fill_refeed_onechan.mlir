//===- memtile_per_fill_refeed_onechan.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-refeed -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// PER-FILL re-send counts where both fills arrive on the SAME CHANNEL, which is
// the shape a real relay has: one producer core ships the slab once per phase
// down one channel, because it has no second port to give.
//
// The two fills then have IDENTICAL descriptors and differ only in their count,
// and `getUniqueBDPattern` used to fold them -- it asks whether two memcpys map
// to equivalent BDs, and read only the offsets/sizes/strides. The chain kept the
// FIRST count and re-sent 12 every fill; the consumer of the second phase waited
// forever for the other 26. Nothing failed at compile time: one BD is a legal
// chain and the write lock still inits to the max.
//
// The count is part of the BD's identity because it IS the BD's lock
// acquire/release value, so the two are now distinguished.

// CHECK: aie.device
// CHECK: %[[MT:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK: %[[WR:.*]] = aie.lock(%[[MT]], {{[0-9]+}}) {init = 38 : i32}
// CHECK: %[[RD:.*]] = aie.lock(%[[MT]], {{[0-9]+}}) {init = 0 : i32}
// CHECK: %[[BUF:.*]] = aie.buffer(%[[MT]]) {air.refeed_count = 38 : i32, air.refeed_per_fill
// CHECK-NOT: aie.buffer(%[[MT]])

// CHECK: aie.memtile_dma(%[[MT]])

// One count-free drain.
// CHECK: aie.dma_start(MM2S, 0, ^[[DRAIN:.*]], ^{{.*}})
// CHECK: ^[[DRAIN]]:
// CHECK: aie.use_lock(%[[RD]], AcquireGreaterEqual, %c1_i32)
// CHECK: aie.next_bd ^[[DRAIN]]

// BOTH fills, in ONE self-looping chain on ONE channel, each RELEASING its own
// count and ACQUIRING THE PREVIOUS FILL'S.
//
// The acquire is not the fill's own count, and that is the whole contract. The
// drain releases the write lock once per re-send, so acquiring N means "N
// re-sends have finished" -- and the N that has to have finished is the one the
// PREVIOUS fill enabled. Acquiring its own 12 here would be satisfied after 12
// of the previous fill's 38 re-sends, and the fill would then overwrite the slab
// with the next phase's data while the other 26 were still being read. Measured
// on qwen3-4b: bit-exact at one decode layer, and 1e-02 off at 36.
//
// With ONE count on the buffer the two numbers coincide, which is why this only
// showed once a slab carried a different count per fill.
// CHECK: aie.dma_start(S2MM, 0, ^[[F1:.*]], ^{{.*}})
// CHECK: ^[[F1]]:
// CHECK: aie.use_lock(%[[WR]], AcquireGreaterEqual, %c38_i32)
// CHECK: aie.dma_bd(%[[BUF]] :
// CHECK: aie.use_lock(%[[RD]], Release, %c12_i32)
// CHECK: aie.next_bd ^[[F2:.*]]
// CHECK: ^[[F2]]:
// CHECK: aie.use_lock(%[[WR]], AcquireGreaterEqual, %c12_i32)
// CHECK: aie.dma_bd(%[[BUF]] :
// CHECK: aie.use_lock(%[[RD]], Release, %c38_i32)
// CHECK: aie.next_bd ^[[F1]]

air.channel @fill [1, 1]
air.channel @drain [1, 1]
func.func @memtile_per_fill_refeed_onechan() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0_i = arith.constant 0 : index
      %c12_i = arith.constant 12 : index
      %c38_i = arith.constant 38 : index
      %t, %l2 = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split} : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      // Same channel, same descriptor, different re-send count.
      air.channel.get @fill[] (%l2[] [] []) : (memref<32xbf16, 1>)
      scf.for %ra = %c0_i to %c12_i step %c1_0 {
        air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      }
      air.channel.get @fill[] (%l2[] [] []) : (memref<32xbf16, 1>)
      scf.for %rb = %c0_i to %c38_i step %c1_0 {
        air.channel.put @drain[] (%l2[] [] []) : (memref<32xbf16, 1>)
      }
      %dd = air.execute {memref.dealloc %l2 : memref<32xbf16, 1>}
      // One producer core, one port, two sends -- it has no second port.
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
