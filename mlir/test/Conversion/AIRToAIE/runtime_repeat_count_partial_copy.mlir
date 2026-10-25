//===- runtime_repeat_count_partial_copy.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics

// [a, b, b] in one run-time loop and [a] in another: the chain [a, b, b, a] is
// not whole copies of a unit, so the loops keep their own tasks, and loops
// with different bounds cannot share a circular chain.

air.channel @c [1, 1]
func.func private @ua(memref<16xbf16, 2>)
func.func private @ub(memref<8xbf16, 2>)
func.func @f(%n: i32, %m: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n, %lm=%m) : i32, i32 {
    air.segment @seg args(%sn=%ln, %sm=%lm) : i32, i32 {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // expected-error@+3 {{'aie.mem' op failed to generate dma bd program}}
      // expected-warning@+2 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+1 {{Enabling herd lock for RTP buffer synchronization}}
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
          }
          scf.for %i = %c0 to %n1 step %c1_h {
            // expected-error@+1 {{in run-time loops with other bounds than those of the transfers before it}}
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
