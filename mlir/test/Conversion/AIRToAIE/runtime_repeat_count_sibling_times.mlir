//===- runtime_repeat_count_sibling_times.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics

// [a, b] per trip of two run-time loops, b alone in a fixed loop of two in
// the first and of three in the second. The second loop's transfers fold
// onto the first loop's BDs, which the chain sends as [a, b, b]; the second
// loop needs [a, b, b, b]. It is rejected.

air.channel @c [1, 1]
func.func private @ka(memref<8xbf16, 2>)
func.func private @vb(memref<8xbf16, 2>)
func.func @sibling_times(%n: i32, %m: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n, %lm=%m) : i32, i32 {
    air.segment @seg args(%sn=%ln, %sm=%lm) : i32, i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %k0 = arith.index_cast %sn : i32 to index
      %k1 = arith.index_cast %sm : i32 to index
      %t0, %a2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0 to %k0 step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
        scf.for %j = %c0 to %c2 step %c1_0 {
          air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
        }
      }
      scf.for %i = %c0 to %k1 step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
        scf.for %j = %c0 to %c3 step %c1_0 {
          air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
        }
      }
      %d2 = air.execute {memref.dealloc %a2 : memref<8xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      // expected-warning@+3 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+2 {{Enabling herd lock for RTP buffer synchronization}}
      // expected-error@+1 {{'aie.mem' op failed to generate dma bd program}}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%hk0=%k0, %hk1=%k1) : index, index attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %c3_h = arith.constant 3 : index
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
            // expected-error@+1 {{repeats a run-time number of times, but also a fixed number of times}}
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            func.call @vb(%b) : (memref<8xbf16, 2>) -> ()
          }
        }
        scf.for %i = %c0_h to %hk1 step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
          func.call @ka(%a) : (memref<8xbf16, 2>) -> ()
          scf.for %j = %c0_h to %c3_h step %c1_h {
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
