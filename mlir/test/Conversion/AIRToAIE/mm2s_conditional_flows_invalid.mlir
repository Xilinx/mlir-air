//===- mm2s_conditional_flows_invalid.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" 2>&1 | FileCheck %s

// Choosing a producer's destination with a branch is rejected, not compiled.
//
// A BD ring advances one BD per transfer no matter which arm the core took, and
// the packet header routes a transfer without selecting which BD carries it. So
// a ring over N conditionally-selected flows is only in step if the arm
// sequence cycles in lockstep with the ring -- nothing in the IR says that, and
// being wrong sends a packet to another flow's destination.
//
// This used to compile silently into a ring the core cannot keep in step with:
// the cyclic order matched the arms, but the DMA entered the ring at the last
// arm's BD, so the very first transfer went to the wrong tile and every one
// after it stayed one slot behind. The S2MM side already refuses this class of
// chain; the producer side now does too.
//
// The fix for a design that wants this is one unconditional loop per flow, as
// in mm2s_flows_program_order.mlir.

// An arm that issues NOTHING is fine here and must not be what trips this --
// the ring only advances when a transfer goes out. What is rejected is arms
// that each issue a different non-empty sequence.

// CHECK: error: {{.*}}compute-tile MM2S channel 0 multiplexes 3 flows (@{{.*}}) over 3 transfers
// CHECK-SAME: control-flow paths deliver different BD sequences
// CHECK: note: flow @{{.*}} on this chain

air.channel @toA [1, 1] {channel_type = "npu_dma_packet"}
air.channel @toB [1, 1] {channel_type = "npu_dma_packet"}
air.channel @toC [1, 1] {channel_type = "npu_dma_packet"}
func.func @switch_egress() {
  %c1 = arith.constant 1 : index
  air.launch (%la, %lb) in (%lc=%c1, %ld=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index

      air.herd @hp tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %t0, %p0 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        scf.for %ph = %c0 to %c3 step %c1i {
          scf.index_switch %ph
          case 0 {
            air.channel.put @toA[] (%p0[] [] []) : (memref<8xbf16, 2>)
            scf.yield
          }
          case 1 {
            air.channel.put @toB[] (%p0[] [] []) : (memref<8xbf16, 2>)
            scf.yield
          }
          default {
            air.channel.put @toC[] (%p0[] [] []) : (memref<8xbf16, 2>)
          }
        }
        %dd0 = air.execute {memref.dealloc %p0 : memref<8xbf16, 2>}
      }

      air.herd @hA tile (%x1, %y1) in (%s1=%c1_0, %r1=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.get @toA[] (%l[] [] []) : (memref<8xbf16, 2>)
        %dda = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hB tile (%x2, %y2) in (%s2=%c1_0, %r2=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.get @toB[] (%l[] [] []) : (memref<8xbf16, 2>)
        %ddb = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hC tile (%x3, %y3) in (%s3=%c1_0, %r3=%c1_0)
            attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.get @toC[] (%l[] [] []) : (memref<8xbf16, 2>)
        %ddc = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
    }
  }
  return
}
