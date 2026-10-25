//===- runtime_repeat_count_fixed_inner_release.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -split-input-file -verify-diagnostics | FileCheck %s

// A core takes [a, b] from one channel per trip of a run-time loop, b in a
// fixed loop of two of its own: the S2MM chain is [a, b, b]. a and b are not
// slots of one rotation, so each has a lock pair of its own with one credit:
// a is released right after its reader, b inside the fixed loop after its.
// CHECK-LABEL: aie.device
// CHECK-DAG: %[[BE:.*]] = aie.lock(%tile_2_3, 4) {init = 1 : i32}
// CHECK-DAG: %[[BF:.*]] = aie.lock(%tile_2_3, 3) {init = 0 : i32}
// CHECK-DAG: %[[AE:.*]] = aie.lock(%tile_2_3, 2) {init = 1 : i32}
// CHECK-DAG: %[[AF:.*]] = aie.lock(%tile_2_3, 1) {init = 0 : i32}
// CHECK: aie.mem(%tile_2_3)
// CHECK: aie.dma_start(S2MM, 0, ^[[BD0:.*]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK-NEXT: aie.use_lock(%[[AE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[A:.*]] : memref<16xbf16, 2>
// CHECK: aie.next_bd ^[[BD1:.*]]
// CHECK: ^[[BD1]]:
// CHECK-NEXT: aie.use_lock(%[[BE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[B:.*]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD2:.*]]
// CHECK: ^[[BD2]]:
// CHECK-NEXT: aie.use_lock(%[[BE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[B]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD0]]
// CHECK: aie.core(%tile_2_3)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[AF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @ka
// CHECK-NEXT: aie.use_lock(%[[AE]], Release
// CHECK-NEXT: scf.for
// CHECK-NEXT: aie.use_lock(%[[BF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @vb
// CHECK-NEXT: aie.use_lock(%[[BE]], Release

air.channel @c [1, 1]
func.func private @ka(memref<16xbf16, 2>)
func.func private @vb(memref<8xbf16, 2>)
func.func @fixed_loop_in_run_time_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %k = arith.index_cast %sn : i32 to index
      %t0, %a2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<16xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
        }
      }
      %d2 = air.execute {memref.dealloc %a2 : memref<16xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      // expected-warning@+2 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+1 {{Enabling herd lock for RTP buffer synchronization}}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%hk=%k) : index attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %t2, %a = air.execute -> (memref<16xbf16, 2>) {
          %alloc = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %alloc : memref<16xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        scf.for %i = %c0_h to %hk step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
          func.call @ka(%a) : (memref<16xbf16, 2>) -> ()
          scf.for %j = %c0_h to %c2_h step %c1_h {
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

// The same with a and b of one type and access pattern, which would make
// them slots of one rotation sharing a counting lock with two credits. But b
// fills twice in a row, and on a shared pair the release of a could admit
// b's second fill before its first is read. So each buffer keeps a pair of
// its own with one credit, as above.
// CHECK-LABEL: aie.device
// CHECK-DAG: %[[BE:.*]] = aie.lock(%tile_2_3, 4) {init = 1 : i32}
// CHECK-DAG: %[[BF:.*]] = aie.lock(%tile_2_3, 3) {init = 0 : i32}
// CHECK-DAG: %[[AE:.*]] = aie.lock(%tile_2_3, 2) {init = 1 : i32}
// CHECK-DAG: %[[AF:.*]] = aie.lock(%tile_2_3, 1) {init = 0 : i32}
// CHECK: aie.mem(%tile_2_3)
// CHECK: aie.dma_start(S2MM, 0, ^[[BD0:.*]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK-NEXT: aie.use_lock(%[[AE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[A:.*]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD1:.*]]
// CHECK: ^[[BD1]]:
// CHECK-NEXT: aie.use_lock(%[[BE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[B:.*]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD2:.*]]
// CHECK: ^[[BD2]]:
// CHECK-NEXT: aie.use_lock(%[[BE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[B]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD0]]
// CHECK: aie.core(%tile_2_3)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[AF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @ka
// CHECK-NEXT: aie.use_lock(%[[AE]], Release
// CHECK-NEXT: scf.for
// CHECK-NEXT: aie.use_lock(%[[BF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @vb
// CHECK-NEXT: aie.use_lock(%[[BE]], Release

air.channel @c [1, 1]
func.func private @ka(memref<8xbf16, 2>)
func.func private @vb(memref<8xbf16, 2>)
func.func @fixed_loop_in_run_time_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %k = arith.index_cast %sn : i32 to index
      %t0, %a2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
        }
      }
      %d2 = air.execute {memref.dealloc %a2 : memref<8xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      // expected-warning@+2 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+1 {{Enabling herd lock for RTP buffer synchronization}}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%hk=%k) : index attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %t2, %a = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        scf.for %i = %c0_h to %hk step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
          func.call @ka(%a) : (memref<8xbf16, 2>) -> ()
          scf.for %j = %c0_h to %c2_h step %c1_h {
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
        }
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// The same gets in two run-time loops with different bounds: each loop takes
// the chain whole, and each releases as above.
// CHECK-LABEL: aie.device
// CHECK-DAG: %[[BE:.*]] = aie.lock(%tile_2_3, 4) {init = 1 : i32}
// CHECK-DAG: %[[BF:.*]] = aie.lock(%tile_2_3, 3) {init = 0 : i32}
// CHECK-DAG: %[[AE:.*]] = aie.lock(%tile_2_3, 2) {init = 1 : i32}
// CHECK-DAG: %[[AF:.*]] = aie.lock(%tile_2_3, 1) {init = 0 : i32}
// CHECK: aie.mem(%tile_2_3)
// CHECK: aie.dma_start(S2MM, 0, ^[[BD0:.*]], ^{{.*}})
// CHECK: ^[[BD0]]:
// CHECK-NEXT: aie.use_lock(%[[AE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[A:.*]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD1:.*]]
// CHECK: ^[[BD1]]:
// CHECK-NEXT: aie.use_lock(%[[BE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[B:.*]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD2:.*]]
// CHECK: ^[[BD2]]:
// CHECK-NEXT: aie.use_lock(%[[BE]], AcquireGreaterEqual
// CHECK-NEXT: aie.dma_bd(%[[B]] : memref<8xbf16, 2>
// CHECK: aie.next_bd ^[[BD0]]
// CHECK: aie.core(%tile_2_3)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[AF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @ka
// CHECK-NEXT: aie.use_lock(%[[AE]], Release
// CHECK-NEXT: scf.for
// CHECK-NEXT: aie.use_lock(%[[BF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @vb
// CHECK-NEXT: aie.use_lock(%[[BE]], Release
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[AF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @ka
// CHECK-NEXT: aie.use_lock(%[[AE]], Release
// CHECK-NEXT: scf.for
// CHECK-NEXT: aie.use_lock(%[[BF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @vb
// CHECK-NEXT: aie.use_lock(%[[BE]], Release

air.channel @c [1, 1]
func.func private @ka(memref<8xbf16, 2>)
func.func private @vb(memref<8xbf16, 2>)
func.func @two_run_time_loops(%n: i32, %m: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n, %lm=%m) : i32, i32 {
    air.segment @seg args(%sn=%ln, %sm=%lm) : i32, i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %k0 = arith.index_cast %sn : i32 to index
      %k1 = arith.index_cast %sm : i32 to index
      %k = arith.addi %k0, %k1 : index
      %t0, %a2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
        }
      }
      %d2 = air.execute {memref.dealloc %a2 : memref<8xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      // expected-warning@+2 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+1 {{Enabling herd lock for RTP buffer synchronization}}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%hk0=%k0, %hk1=%k1) : index, index attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %t2, %a = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        scf.for %i = %c0_h to %hk0 step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
          func.call @ka(%a) : (memref<8xbf16, 2>) -> ()
          scf.for %j = %c0_h to %c2_h step %c1_h {
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
        }
        scf.for %i = %c0_h to %hk1 step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
          func.call @ka(%a) : (memref<8xbf16, 2>) -> ()
          scf.for %j = %c0_h to %c2_h step %c1_h {
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
        }
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// As the second case, with a read again after the fixed loop. On a shared
// pair with two credits the second release in the fixed loop would let the
// DMA refill a before that read. With a pair each, a is released after its
// last read and b inside the fixed loop.
// CHECK-LABEL: aie.device
// CHECK-DAG: %[[BE:.*]] = aie.lock(%tile_2_3, 4) {init = 1 : i32}
// CHECK-DAG: %[[BF:.*]] = aie.lock(%tile_2_3, 3) {init = 0 : i32}
// CHECK-DAG: %[[AE:.*]] = aie.lock(%tile_2_3, 2) {init = 1 : i32}
// CHECK-DAG: %[[AF:.*]] = aie.lock(%tile_2_3, 1) {init = 0 : i32}
// CHECK: aie.core(%tile_2_3)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[AF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @ka
// CHECK-NEXT: scf.for
// CHECK-NEXT: aie.use_lock(%[[BF]], AcquireGreaterEqual
// CHECK-NEXT: func.call @vb
// CHECK-NEXT: aie.use_lock(%[[BE]], Release
// CHECK-NEXT: }
// CHECK-NEXT: func.call @ka
// CHECK-NEXT: aie.use_lock(%[[AE]], Release

air.channel @c [1, 1]
func.func private @ka(memref<8xbf16, 2>)
func.func private @vb(memref<8xbf16, 2>)
func.func @read_after_fixed_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %k = arith.index_cast %sn : i32 to index
      %t0, %a2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
        }
      }
      %d2 = air.execute {memref.dealloc %a2 : memref<8xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      // expected-warning@+2 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+1 {{Enabling herd lock for RTP buffer synchronization}}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%hk=%k) : index attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %t2, %a = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        scf.for %i = %c0_h to %hk step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
          func.call @ka(%a) : (memref<8xbf16, 2>) -> ()
          scf.for %j = %c0_h to %c2_h step %c1_h {
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
          func.call @ka(%a) : (memref<8xbf16, 2>) -> ()
        }
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}
