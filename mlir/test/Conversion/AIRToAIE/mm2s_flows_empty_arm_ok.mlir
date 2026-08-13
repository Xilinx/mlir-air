//===- mm2s_flows_empty_arm_ok.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// A producer arm that issues NOTHING is not a hole. Two flows share one MM2S
// and both puts sit in the SAME arm of a mode switch; the other arm sends
// neither. That must compile.
//
// The consumer-side rule says every arm has to present the same BD sequence,
// because a receiver that skips an arrival still gets the packet and lands it
// on whatever BD the pointer sits on. A producer is not symmetric: taking the
// silent arm issues no transfer, so the ring does not advance and stays
// aligned. What still has to hold -- and does here -- is that the arm which
// does transmit covers a whole ring cycle.
//
// Reusing the consumer rule verbatim rejected this shape, which is the KV-cache
// append in the fused decode designs (both appends in the decode arm, none in
// the vocab arm). See mm2s_conditional_flows_invalid.mlir for the shape that is
// genuinely unsafe: arms issuing DIFFERENT non-empty sequences.

// CHECK: aie.dma_start(MM2S
// CHECK-COUNT-2: aie.packet_flow(
// CHECK-NOT: aie.packet_flow(

air.channel @appendK [1, 1] {channel_type = "npu_dma_packet"}
air.channel @appendV [1, 1] {channel_type = "npu_dma_packet"}
func.func @empty_arm_producer() {
  %c1 = arith.constant 1 : index
  air.launch (%la, %lb) in (%lc=%c1, %ld=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index

      air.herd @hp tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %pk = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t1, %pv = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        scf.for %m = %c0 to %c2 step %c1i {
          scf.index_switch %m
          case 0 {
            // decode arm: append this token's K and V.
            air.channel.put @appendK[] (%pk[] [] []) : (memref<8xbf16, 2>)
            air.channel.put @appendV[] (%pv[] [] []) : (memref<8xbf16, 2>)
            scf.yield
          }
          default {
            // vocab arm: no appends at all.
          }
        }
        %dd0 = air.execute {memref.dealloc %pk : memref<8xbf16, 2>}
      }

      air.herd @hK tile (%x1, %y1) in (%s1=%c1_0, %r1=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.get @appendK[] (%l[] [] []) : (memref<8xbf16, 2>)
        %ddk = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hV tile (%x2, %y2) in (%s2=%c1_0, %r2=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.get @appendV[] (%l[] [] []) : (memref<8xbf16, 2>)
        %ddv = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
    }
  }
  return
}
