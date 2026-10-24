//===- runtime_repeat_count_fixed_inner.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -split-input-file -verify-diagnostics | FileCheck %s

// Two transfers repeat 2 and 3 times within each trip of a run-time loop, each
// alone in its fixed loop. The circular BD chain takes the first twice and the
// second three times, then starts over for the next trip.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[BB:.*]], ^
// CHECK: ^[[BB]]:
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[BB]]

air.channel @r0 [1, 1]
func.func @runtime_repeat_with_fixed_inner(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        scf.for %j = %c0 to %c2 step %c1_0 {
          air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
        scf.for %j = %c0 to %c3 step %c1_0 {
          air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<16xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}


// -----

// Two transfers in one fixed loop within a run-time loop: each trip runs them
// in turn twice, which copies of each BD in a row cannot express.
// expected-error@unknown {{'aie.memtile_dma' op failed to generate dma bd program}}

air.channel @r0 [1, 1]
func.func @two_in_fixed_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          // expected-error@+1 {{repeats a run-time number of times, but also a fixed number of times}}
          air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
          air.channel.put @r0[] (%l2[%c4] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<16xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// Each transfer is in a run-time loop inside a fixed loop of two trips. The
// fixed loops repeat both transfers alike, which the circular chain does
// anyway: one BD each.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[BB:.*]], ^
// CHECK: ^[[BB]]:
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[BB2:.*]]
// CHECK: ^[[BB2]]:
// CHECK: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[BB]]

air.channel @r0 [1, 1]
func.func @outer_fixed_alike(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      scf.for %o = %c0 to %c2 step %c1_0 {
        scf.for %i = %c0 to %k step %c1_0 {
          air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
      }
      scf.for %o = %c0 to %c2 step %c1_0 {
        scf.for %i = %c0 to %k step %c1_0 {
          air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<16xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}


// -----

// Only the first transfer's run-time loop is inside a fixed loop: the first
// is sent twice as often as the second, which one circular chain cannot do.

air.channel @r0 [1, 1]
func.func @outer_fixed_unlike(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      scf.for %o = %c0 to %c2 step %c1_0 {
        scf.for %i = %c0 to %k step %c1_0 {
          // expected-error@+1 {{repeats a run-time number of times, but also a fixed number of times}}
          air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        }
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<16xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// A transfer alone in two nested fixed loops (2 and 3 trips) within a run-time
// loop, after a transfer sent once per trip. The circular BD chain takes the
// first once and the second 2 * 3 = 6 times in a row, then starts over.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[BB:.*]], ^
// CHECK: ^[[BB]]:
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK-COUNT-6: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[BB]]
// CHECK-NOT: aie.dma_bd
// CHECK: aie.end

air.channel @r0 [1, 1]
func.func @alone_in_two_fixed_loops(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @r0[] (%l2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          scf.for %l = %c0 to %c3 step %c1_0 {
            air.channel.put @r0[] (%l2[%c8] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
          }
        }
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<16xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 7 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// The same shape on the receiving side: a core's gets into one L1 buffer, the
// second alone in two nested fixed loops within a run-time loop. The tile
// DMA's S2MM chain takes the first once and the second six times in a row.
// CHECK-LABEL: aie.device
// CHECK: aie.mem(
// CHECK: aie.dma_start(S2MM, 0, ^[[BB:.*]], ^
// CHECK: ^[[BB]]:
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 8)
// CHECK-COUNT-6: aie.dma_bd({{.*}} offset = 8 len = 8)
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[BB]]
// CHECK-NOT: aie.dma_bd
// CHECK: aie.end

air.channel @r0 [1, 1]
func.func @core_get_alone_in_two_fixed_loops(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k = arith.index_cast %sn : i32 to index
      %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @r0[] (%l2[] [] []) : (memref<8xbf16, 1>)
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<8xbf16, 1>}
      // The run-time trip count reaches the core through a run-time parameter.
      // expected-warning@+2 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+1 {{Enabling herd lock for RTP buffer synchronization}}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) args(%hk=%k) : index attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %c3_h = arith.constant 3 : index
        %c8_h = arith.constant 8 : index
        %tok0, %l1a = air.execute -> (memref<16xbf16, 2>) {
          %aa = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %aa : memref<16xbf16, 2>
        }
        scf.for %i = %c0_h to %hk step %c1_h {
          air.channel.get @r0[] (%l1a[%c0_h] [%c8_h] [%c1_h]) : (memref<16xbf16, 2>)
          scf.for %j = %c0_h to %c2_h step %c1_h {
            scf.for %l = %c0_h to %c3_h step %c1_h {
              air.channel.get @r0[] (%l1a[%c8_h] [%c8_h] [%c1_h]) : (memref<16xbf16, 2>)
            }
          }
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<16xbf16, 2>}
      }
    }
  }
  return
}
