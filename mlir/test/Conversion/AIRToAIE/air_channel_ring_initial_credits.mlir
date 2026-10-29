//===- air_channel_ring_initial_credits.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics

// A core takes [a, a, b] from one channel, a and b slots of one rotation that
// share a counting lock with two credits. The two initial credits let the DMA
// fill a twice before the core reads the first fill; no release placement
// can prevent that. It is rejected.

air.channel @c [1, 1]
func.func private @use(memref<8xbf16, 2>)
func.func @a_a_b() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %t0, %a2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
      %d2 = air.execute {memref.dealloc %a2 : memref<8xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %t2, %a = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        // expected-error@+2 {{shares a counting lock with the other buffers of its rotation}}
        // expected-error@+1 {{failed to allocate core locks}}
        air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
        func.call @use(%a) : (memref<8xbf16, 2>) -> ()
        air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
        func.call @use(%a) : (memref<8xbf16, 2>) -> ()
        air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
        func.call @use(%b) : (memref<8xbf16, 2>) -> ()
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}
