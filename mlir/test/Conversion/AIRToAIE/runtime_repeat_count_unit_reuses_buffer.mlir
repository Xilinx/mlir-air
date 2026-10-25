//===- runtime_repeat_count_unit_reuses_buffer.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// Two run-time loops with different bounds each take [a, b, b] from one
// channel, b twice per trip. Each takes the chain whole, so the core's S2MM
// is one circular chain of three BDs. Its two slots share one counting lock:
// releasing a lets the DMA refill b, so a is released only after b's first
// reader, in both loops.
// CHECK-LABEL: aie.mem(%tile_2_4)
// CHECK: aie.dma_start(S2MM, 0, ^[[BD0:bb[0-9]+]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK: aie.dma_bd(%[[A:.*]] : memref<16xbf16, 2>
// CHECK: aie.next_bd ^[[BD1:bb[0-9]+]]
// CHECK: ^[[BD1]]:
// CHECK: aie.dma_bd(%[[B:.*]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD2:bb[0-9]+]]
// CHECK: ^[[BD2]]:
// CHECK: aie.dma_bd(%[[B]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD0]]
// CHECK-LABEL: aie.core(%tile_2_4)
// CHECK: scf.for
// CHECK: func.call @ka
// CHECK-NOT: Release
// CHECK: func.call @vb
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK: scf.for
// CHECK: func.call @ka
// CHECK-NOT: Release
// CHECK: func.call @vb
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})

air.channel @c [1, 1]
func.func private @ka(memref<16xbf16, 2>)
func.func private @vb(memref<8xbf16, 2>)
func.func @two_loops(%n: i32, %m: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n, %lm=%m) : i32, i32 {
    air.segment @seg args(%sn=%ln, %sm=%lm) : i32, i32 {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2) args(%hn=%sn, %hm=%sm) : i32, i32 attributes {link_with = "use.o"} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %n0 = arith.index_cast %hn : i32 to index
        %n1 = arith.index_cast %hm : i32 to index
        %t0, %a = air.execute -> (memref<16xbf16, 2>) {
          %alloc = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %alloc : memref<16xbf16, 2>
        }
        %t1, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %is0 = arith.cmpi eq, %ty, %c0 : index
        scf.if %is0 {
          %nt = arith.addi %n0, %n1 : index
          scf.for %i = %c0 to %nt step %c1_h {
            air.channel.put @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            air.channel.put @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            air.channel.put @c[] (%b[] [] []) : (memref<8xbf16, 2>)
          }
        } else {
          scf.for %i = %c0 to %n0 step %c1_h {
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ka(%a) : (memref<16xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
          scf.for %i = %c0 to %n1 step %c1_h {
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ka(%a) : (memref<16xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
        }
        %d0 = air.execute {memref.dealloc %a : memref<16xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// [a, b, a] in each of two such loops: the unit is the whole three gets, and
// with two slots each release follows the reader of the buffer it lets the
// DMA refill: the first a's, then the second a's and b's.
// CHECK-LABEL: aie.mem(%tile_2_4)
// CHECK: aie.dma_start(S2MM, 0, ^[[BD0:bb[0-9]+]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK: aie.dma_bd(%[[A:.*]] : memref<16xbf16, 2>
// CHECK-NOT: aie.next_bd ^[[BD0]]
// CHECK: aie.dma_bd(%[[B:.*]] : memref<8xbf16, 2>
// CHECK-NOT: aie.next_bd ^[[BD0]]
// CHECK: aie.dma_bd(%[[A]] : memref<16xbf16, 2>
// CHECK-NOT: aie.dma_bd
// CHECK: aie.next_bd ^[[BD0]]
// CHECK-LABEL: aie.core(%tile_2_4)
// CHECK: scf.for
// CHECK: func.call @ua
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK: func.call @ub
// CHECK-NOT: Release
// CHECK: func.call @ua
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK: scf.for
// CHECK: func.call @ua
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK: func.call @ub
// CHECK-NOT: Release
// CHECK: func.call @ua
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})

air.channel @c [1, 1]
func.func private @ua(memref<16xbf16, 2>)
func.func private @ub(memref<8xbf16, 2>)
func.func @unit_aba(%n: i32, %m: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n, %lm=%m) : i32, i32 {
    air.segment @seg args(%sn=%ln, %sm=%lm) : i32, i32 {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2) args(%hn=%sn, %hm=%sm) : i32, i32 attributes {link_with = "u.o"} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %n0 = arith.index_cast %hn : i32 to index
        %n1 = arith.index_cast %hm : i32 to index
        %t0, %a = air.execute -> (memref<16xbf16, 2>) {
          %alloc = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %alloc : memref<16xbf16, 2>
        }
        %t1, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %is0 = arith.cmpi eq, %ty, %c0 : index
        scf.if %is0 {
          %nt = arith.addi %n0, %n1 : index
          scf.for %i = %c0 to %nt step %c1_h {
            air.channel.put @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            air.channel.put @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            air.channel.put @c[] (%a[] [] []) : (memref<16xbf16, 2>)
          }
        } else {
          scf.for %i = %c0 to %n0 step %c1_h {
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ua(%a) : (memref<16xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @ub(%b) : (memref<8xbf16, 2>) -> ()
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ua(%a) : (memref<16xbf16, 2>) -> ()
          }
          scf.for %i = %c0 to %n1 step %c1_h {
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ua(%a) : (memref<16xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @ub(%b) : (memref<8xbf16, 2>) -> ()
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ua(%a) : (memref<16xbf16, 2>) -> ()
          }
        }
        %d0 = air.execute {memref.dealloc %a : memref<16xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// [a, b, b, a] in one loop is not whole copies of a shorter unit: it keeps
// one BD per get.
// CHECK-LABEL: aie.mem(%tile_2_4)
// CHECK: aie.dma_start(S2MM, 0, ^[[BD0:bb[0-9]+]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK: aie.dma_bd(%{{.*}} : memref<16xbf16, 2>
// CHECK-NOT: aie.next_bd ^[[BD0]]
// CHECK: aie.dma_bd(%{{.*}} : memref<8xbf16, 2>
// CHECK-NOT: aie.next_bd ^[[BD0]]
// CHECK: aie.dma_bd(%{{.*}} : memref<8xbf16, 2>
// CHECK-NOT: aie.next_bd ^[[BD0]]
// CHECK: aie.dma_bd(%{{.*}} : memref<16xbf16, 2>
// CHECK-NOT: aie.dma_bd
// CHECK: aie.next_bd ^[[BD0]]

air.channel @c [1, 1]
func.func private @ua(memref<16xbf16, 2>)
func.func private @ub(memref<8xbf16, 2>)
func.func @not_whole_copies(%n: i32, %m: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n, %lm=%m) : i32, i32 {
    air.segment @seg args(%sn=%ln, %sm=%lm) : i32, i32 {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2) args(%hn=%sn, %hm=%sm) : i32, i32 attributes {link_with = "u.o"} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %n0 = arith.index_cast %hn : i32 to index
        %n1 = arith.index_cast %hm : i32 to index
        %t0, %a = air.execute -> (memref<16xbf16, 2>) {
          %alloc = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %alloc : memref<16xbf16, 2>
        }
        %t1, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %is0 = arith.cmpi eq, %ty, %c0 : index
        scf.if %is0 {
          %nt = arith.addi %n0, %n1 : index
          scf.for %i = %c0 to %nt step %c1_h {
            air.channel.put @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            air.channel.put @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            air.channel.put @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            air.channel.put @c[] (%a[] [] []) : (memref<16xbf16, 2>)
          }
        } else {
          scf.for %i = %c0 to %n0 step %c1_h {
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ua(%a) : (memref<16xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @ub(%b) : (memref<8xbf16, 2>) -> ()
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @ub(%b) : (memref<8xbf16, 2>) -> ()
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            func.call @ua(%a) : (memref<16xbf16, 2>) -> ()
          }
        }
        %d0 = air.execute {memref.dealloc %a : memref<16xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// A re-fed producer whose puts are whole copies of [x, z, z]: each transfer
// is sent twice per pass, in program order, x x z z z z.
// CHECK-LABEL: aie.buffer(%tile_2_3) {{.*}} : memref<32xbf16, 2>
// CHECK: aie.mem(%tile_2_3)
// CHECK: aie.dma_start(MM2S, 0, ^[[BD0:bb[0-9]+]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK-NOT: aie.dma_bd(%{{.*}} : memref<4xbf16, 2>
// CHECK-COUNT-2: aie.dma_bd(%{{.*}} : memref<32xbf16, 2>
// CHECK-NOT: aie.dma_bd(%{{.*}} : memref<32xbf16, 2>
// CHECK-COUNT-4: aie.dma_bd(%{{.*}} : memref<4xbf16, 2>
// CHECK-NOT: aie.dma_bd
// CHECK: aie.next_bd ^[[BD0]]

air.channel @c [1, 1] {air.refeed_count = 2 : i32}
func.func @refeed_unit() {
  %c1 = arith.constant 1 : index
  air.launch (%x0, %y0) in (%sx=%c1, %sy=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c12 = arith.constant 12 : index
        %t0, %x = air.execute -> (memref<32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %alloc : memref<32xbf16, 2>
        }
        %t1, %z = air.execute -> (memref<4xbf16, 2>) {
          %alloc = memref.alloc() : memref<4xbf16, 2>
          air.execute_terminator %alloc : memref<4xbf16, 2>
        }
        %is0 = arith.cmpi eq, %ty, %c0 : index
        scf.if %is0 {
          air.channel.put @c[] (%x[] [] []) : (memref<32xbf16, 2>)
          air.channel.put @c[] (%z[] [] []) : (memref<4xbf16, 2>)
          air.channel.put @c[] (%z[] [] []) : (memref<4xbf16, 2>)
          air.channel.put @c[] (%x[] [] []) : (memref<32xbf16, 2>)
          air.channel.put @c[] (%z[] [] []) : (memref<4xbf16, 2>)
          air.channel.put @c[] (%z[] [] []) : (memref<4xbf16, 2>)
        } else {
          scf.for %i = %c0 to %c12 step %c1_h {
            air.channel.get @c[] (%z[] [] []) : (memref<4xbf16, 2>)
          }
        }
        %d0 = air.execute {memref.dealloc %x : memref<32xbf16, 2>}
        %d1 = air.execute {memref.dealloc %z : memref<4xbf16, 2>}
      }
    }
  }
  return
}
