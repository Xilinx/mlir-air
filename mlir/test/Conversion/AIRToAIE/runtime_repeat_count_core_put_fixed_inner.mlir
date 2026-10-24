//===- runtime_repeat_count_core_put_fixed_inner.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics

// A core sends [a, b, b] per trip of a run-time loop, b from a fixed loop of
// two of its own. The core's buffers on the channel share one counting lock
// pair, and b's sends would release it inside the fixed loop ahead of a's
// release after it, out of the chain's order. It is rejected.

air.channel @c [1, 1]
func.func @core_put_in_fixed_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // expected-warning@+3 {{Herd RTP buffer allocated but herd lock disabled}}
      // expected-warning@+2 {{Enabling herd lock for RTP buffer synchronization}}
      // expected-error@+1 {{'aie.mem' op failed to generate dma bd program}}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2) args(%hn=%sn) : i32 {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %n0 = arith.index_cast %hn : i32 to index
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
          scf.for %i = %c0 to %n0 step %c1_h {
            air.channel.put @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            scf.for %j = %c0 to %c2_h step %c1_h {
              // expected-error@+1 {{is sent several times in a row per trip from a core}}
              air.channel.put @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            }
          }
        } else {
          scf.for %i = %c0 to %n0 step %c1_h {
            air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
            air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
          }
        }
        %d0 = air.execute {memref.dealloc %a : memref<16xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}
