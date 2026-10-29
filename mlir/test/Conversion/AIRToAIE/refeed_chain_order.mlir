//===- refeed_chain_order.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 use-lock-race-condition-fix-v2=true" --split-input-file | FileCheck %s

// A transfer re-fed N times (air.refeed_count, on the memtile buffer or the
// core put) is sent N times per fill by a count-free MM2S chain. When such
// transfers share one circular chain, it must send what each logical flow (a
// FIFO) needs per fill, in order.

// Two buffers re-fed 4 times each on ONE flow: the flow wants x^4 z^4. A
// one-BD-per-buffer chain would send (x z)^4, out of order, and deadlock when
// z is only filled after x has been consumed. Expect one chain on one channel
// sending x four times, then z four times.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-COUNT-3: aie.dma_bd(%[[X]] :
// CHECK-COUNT-4: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]
// CHECK-NOT: aie.dma_start(MM2S, 1,
// CHECK: aie.end

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @one_flow_equal_counts(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %t2, %buf = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c8 step %c1_h {
          air.channel.get @out[] (%buf[] [] []) : (memref<64xbf16, 2>)
        }
        air.channel.put @done[] (%buf[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %buf : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// Unequal counts on one flow: x^2 z^3, still one pass per fill. The two puts
// spell the one slot of @out differently, [] and [0].
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-COUNT-1: aie.dma_bd(%[[X]] :
// CHECK-COUNT-3: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]
// CHECK-NOT: aie.dma_start(MM2S, 1,
// CHECK: aie.end

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @one_flow_unequal_counts(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %c0_s = arith.constant 0 : index
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 2 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 3 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[%c0_s] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c5 = arith.constant 5 : index
        %t2, %buf = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c5 step %c1_h {
          air.channel.get @out[] (%buf[] [] []) : (memref<64xbf16, 2>)
        }
        air.channel.put @done[] (%buf[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %buf : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// Two flows re-fed at the same rate are each a FIFO of one transfer: one BD
// each, alternating, sends x^4 and z^4 per fill. Expect one BD each.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B1:bb[0-9]+]]
// CHECK: ^[[B1]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]
// CHECK: aie.end

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @outX [1] {channel_type = "npu_dma_packet"}
air.channel @outZ [1] {channel_type = "npu_dma_packet"}
func.func @two_flows_equal_counts(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outX[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outZ[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t2, %bx = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        %t3, %bz = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c4 step %c1_h {
          air.channel.get @outX[] (%bx[] [] []) : (memref<64xbf16, 2>)
          air.channel.get @outZ[] (%bz[] [] []) : (memref<64xbf16, 2>)
        }
        air.channel.put @done[] (%bx[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %bx : memref<64xbf16, 2> }
        %d3 = air.execute { memref.dealloc %bz : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// Two flows at different rates, x re-fed 4 times and z 2 times. One BD each
// would ask z for a send per send of x, which z cannot supply. Two passes per
// fill of x x z deliver both rates.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-COUNT-1: aie.dma_bd(%[[X]] :
// CHECK-COUNT-1: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]
// CHECK: aie.end

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @outX [1] {channel_type = "npu_dma_packet"}
air.channel @outZ [1] {channel_type = "npu_dma_packet"}
func.func @two_flows_unequal_counts(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 2 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outX[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outZ[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t2, %bx = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        %t3, %bz = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c2 step %c1_h {
          air.channel.get @outX[] (%bx[] [] []) : (memref<64xbf16, 2>)
          air.channel.get @outX[] (%bx[] [] []) : (memref<64xbf16, 2>)
          air.channel.get @outZ[] (%bz[] [] []) : (memref<64xbf16, 2>)
        }
        air.channel.put @done[] (%bx[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %bx : memref<64xbf16, 2> }
        %d3 = air.execute { memref.dealloc %bz : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// Without air.refeed_count each buffer gets one BD.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B1:bb[0-9]+]]
// CHECK: ^[[B1]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]
// CHECK: aie.end

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @no_refeed(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t2, %buf = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c2 step %c1_h {
          air.channel.get @out[] (%buf[] [] []) : (memref<64xbf16, 2>)
        }
        air.channel.put @done[] (%buf[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %buf : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// A memtile buffer drained by two ops carries their summed count, which does
// not say how many sends belong to each op. Expect one BD per op.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] : {{.*}} offset = 0 len = 32
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B1:bb[0-9]+]]
// CHECK: ^[[B1]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X]] : {{.*}} offset = 32 len = 32
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B2:bb[0-9]+]]
// CHECK: ^[[B2]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @buffer_drained_twice(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %c0_s = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c32_s = arith.constant 32 : index
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 3 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 2 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[%c0_s] [%c32_s] [%c1_s]) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[%c32_s] [%c32_s] [%c1_s]) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c5 = arith.constant 5 : index
        %t2, %buf = air.execute -> (memref<32xbf16, 2>) {
          %allocated = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %allocated : memref<32xbf16, 2>
        }
        scf.for %i = %c0 to %c5 step %c1_h {
          air.channel.get @out[] (%buf[] [] []) : (memref<32xbf16, 2>)
        }
        air.channel.put @done[] (%buf[] [] []) : (memref<32xbf16, 2>)
        %d2 = air.execute { memref.dealloc %buf : memref<32xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[0] [32] [1]) : (memref<64xbf16>)
  }
  return
}

// -----

// A core producer carries the count on the put. Two puts on one flow, re-fed
// 2 and 3 times, drain as x^2 z^3.
// CHECK-LABEL: aie.device
// CHECK: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-COUNT-1: aie.dma_bd(%[[X]] :
// CHECK-COUNT-3: aie.dma_bd(%[[Z:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]

air.channel @c [1, 1]
func.func @core_one_flow() {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c5 = arith.constant 5 : index
        %cmp = arith.cmpi eq, %ty, %c0 : index
        scf.if %cmp {
          %x = memref.alloc() : memref<32xbf16, 2>
          %z = memref.alloc() : memref<32xbf16, 2>
          air.channel.put @c[] (%x[] [] []) {air.refeed_count = 2 : i32} : (memref<32xbf16, 2>)
          air.channel.put @c[] (%z[] [] []) {air.refeed_count = 3 : i32} : (memref<32xbf16, 2>)
          memref.dealloc %x : memref<32xbf16, 2>
          memref.dealloc %z : memref<32xbf16, 2>
        } else {
          %y = memref.alloc() : memref<32xbf16, 2>
          scf.for %i = %c0 to %c5 step %c1_h {
            air.channel.get @c[] (%y[] [] []) : (memref<32xbf16, 2>)
          }
          memref.dealloc %y : memref<32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// The emitter folds the chain [x, w, w, w, w] to [x, w], so the w BD stands
// for four transfers. x is re-fed 4 times, so x and w are in step at one BD
// each.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B1:bb[0-9]+]]
// CHECK: ^[[B1]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[W:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]

air.channel @fillX [1]
air.channel @fillY [1]
air.channel @done [1]
air.channel @outX [1] {channel_type = "npu_dma_packet"}
air.channel @outY [1] {channel_type = "npu_dma_packet"}
func.func @refeed_next_to_folded_suffix(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fillX[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fillY[] (%bb[] [] []) : (memref<64xbf16>)
    air.channel.put @fillY[] (%bb[] [] []) : (memref<64xbf16>)
    air.channel.put @fillY[] (%bb[] [] []) : (memref<64xbf16>)
    air.channel.put @fillY[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %w = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fillX[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outX[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fillY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fillY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fillY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fillY[] (%w[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @outY[] (%w[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %w : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t2, %bx = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        %t3, %bw = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        scf.for %i = %c0 to %c4 step %c1_h {
          air.channel.get @outX[] (%bx[] [] []) : (memref<64xbf16, 2>)
          air.channel.get @outY[] (%bw[] [] []) : (memref<64xbf16, 2>)
        }
        air.channel.put @done[] (%bx[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %bx : memref<64xbf16, 2> }
        %d3 = air.execute { memref.dealloc %bw : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// A unit that sends one buffer twice: the core sends [x, w, w] twice, x
// re-fed 2 times. The chain folds to the unit [x, w, w], and each of its
// slots stands for two transfers, one per repetition: x is sent twice per
// pass and each w once, x^2 w w on the one flow.
// CHECK-LABEL: aie.device
// CHECK: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_start(MM2S, 0, ^[[B0:bb[0-9]+]],
// CHECK: ^[[B0]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B1:bb[0-9]+]]
// CHECK: ^[[B1]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[X]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B2:bb[0-9]+]]
// CHECK: ^[[B2]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[W:buf[0-9]+]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B3:bb[0-9]+]]
// CHECK: ^[[B3]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[W]] :
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[B0]]

air.channel @c [1, 1]
func.func @core_unit_sends_buffer_twice() {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %cmp = arith.cmpi eq, %ty, %c0 : index
        scf.if %cmp {
          %x = memref.alloc() : memref<32xbf16, 2>
          %w = memref.alloc() : memref<32xbf16, 2>
          air.channel.put @c[] (%x[] [] []) {air.refeed_count = 2 : i32} : (memref<32xbf16, 2>)
          air.channel.put @c[] (%w[] [] []) : (memref<32xbf16, 2>)
          air.channel.put @c[] (%w[] [] []) : (memref<32xbf16, 2>)
          air.channel.put @c[] (%x[] [] []) {air.refeed_count = 2 : i32} : (memref<32xbf16, 2>)
          air.channel.put @c[] (%w[] [] []) : (memref<32xbf16, 2>)
          air.channel.put @c[] (%w[] [] []) : (memref<32xbf16, 2>)
          memref.dealloc %x : memref<32xbf16, 2>
          memref.dealloc %w : memref<32xbf16, 2>
        } else {
          %y = memref.alloc() : memref<32xbf16, 2>
          scf.for %i = %c0 to %c8 step %c1_h {
            air.channel.get @c[] (%y[] [] []) : (memref<32xbf16, 2>)
          }
          memref.dealloc %y : memref<32xbf16, 2>
        }
      }
    }
  }
  return
}
