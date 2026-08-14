//===- mm2s_flows_program_order.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Several transfer loops sharing one MM2S become one BD task PER LOOP, in
// program order -- even when two of them have the same trip count.
//
// BD tasks used to be bucketed by trip count alone, so two loops that happened
// to share a count were spliced into a single task and ran together at the
// position of the first. Here phases 1 and 3 both iterate 4 times, so the
// phase-3 transfers were hoisted ahead of the 32 phase-2 transfers they must
// follow: the emitted order was [rope x6, rms x4 + rms x4, glu x32] instead of
// [rope x6, rms x4, glu x32, rms x4]. The counts were right and the order was
// not, which on device means the consumer receives a round of data before the
// producer has computed it.
//
// Phases 1 and 3 also share a CHANNEL, so they must share a packet id: one
// logical flow reached from two points in the schedule.

// CHECK: aie.dma_start(MM2S, 0, {{[^)]*}}repeat_count = 5)
// CHECK: aie.dma_bd({{.*}}pkt_id = [[ROPE:[0-9]+]]>, task_id = 0
// CHECK: aie.dma_start(MM2S, 0, {{[^)]*}}repeat_count = 3)
// CHECK: aie.dma_bd({{.*}}pkt_id = [[RMS:[0-9]+]]>, task_id = 1
// CHECK: aie.dma_start(MM2S, 0, {{[^)]*}}repeat_count = 31)
// CHECK: aie.dma_bd({{.*}}pkt_id = [[GLU:[0-9]+]]>, task_id = 2
// CHECK: aie.dma_start(MM2S, 0, {{[^)]*}}repeat_count = 3)
// CHECK: aie.dma_bd({{.*}}pkt_id = [[RMS]]>, task_id = 3

air.channel @toRope [1, 1] {channel_type = "npu_dma_packet"}
air.channel @toRms  [1, 1] {channel_type = "npu_dma_packet"}
air.channel @toGlu  [1, 1] {channel_type = "npu_dma_packet"}
func.func @perphase_loops() {
  %c1 = arith.constant 1 : index
  air.launch (%la, %lb) in (%lc=%c1, %ld=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index

      air.herd @hp tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c6 = arith.constant 6 : index
        %c32 = arith.constant 32 : index
        %t0, %p0 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t1, %p1 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t2, %p2 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t3, %p3 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        scf.for %r0 = %c0 to %c6 step %c1i {
          air.channel.put @toRope[] (%p0[] [] []) : (memref<8xbf16, 2>)
        }
        scf.for %r1 = %c0 to %c4 step %c1i {
          air.channel.put @toRms[] (%p1[] [] []) : (memref<8xbf16, 2>)
        }
        scf.for %r2 = %c0 to %c32 step %c1i {
          air.channel.put @toGlu[] (%p2[] [] []) : (memref<8xbf16, 2>)
        }
        scf.for %r3 = %c0 to %c4 step %c1i {
          air.channel.put @toRms[] (%p3[] [] []) : (memref<8xbf16, 2>)
        }
        %dd0 = air.execute {memref.dealloc %p0 : memref<8xbf16, 2>}
      }

      air.herd @hRope tile (%x1, %y1) in (%s1=%c1_0, %r1h=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c6 = arith.constant 6 : index
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        scf.for %r = %c0 to %c6 step %c1i {
          air.channel.get @toRope[] (%l[] [] []) : (memref<8xbf16, 2>)
        }
        %dda = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hRms tile (%x2, %y2) in (%s2=%c1_0, %r2h=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t2, %l2 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        scf.for %r = %c0 to %c4 step %c1i {
          air.channel.get @toRms[] (%l[] [] []) : (memref<8xbf16, 2>)
        }
        scf.for %rr = %c0 to %c4 step %c1i {
          air.channel.get @toRms[] (%l2[] [] []) : (memref<8xbf16, 2>)
        }
        %ddb = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hGlu tile (%x3, %y3) in (%s3=%c1_0, %r3h=%c1_0)
            attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        scf.for %r = %c0 to %c32 step %c1i {
          air.channel.get @toGlu[] (%l[] [] []) : (memref<8xbf16, 2>)
        }
        %ddc = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
    }
  }
  return
}
