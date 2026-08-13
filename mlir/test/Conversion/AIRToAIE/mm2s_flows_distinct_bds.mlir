//===- mm2s_flows_distinct_bds.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Transfers on DIFFERENT channels never fold onto one BD, however alike they
// look. All three puts here send the same buffer with the same access pattern,
// so BD equivalence -- which compared memref plus offsets/sizes/strides -- used
// to call them interchangeable and keep a single BD. The three flows were still
// emitted, but only the surviving BD's id was ever put on the wire, so every
// packet went to that one destination and the other two consumers waited
// forever. A BD carries its flow's routing, so the channel is part of its
// identity.

// Three BDs on one MM2S, three distinct ids, one per flow.
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.dma_bd({{.*}}pkt_id = [[A:[0-9]+]]>
// CHECK: aie.dma_bd({{.*}}pkt_id = [[B:[0-9]+]]>
// CHECK-NOT: pkt_id = [[A]]>
// CHECK: aie.dma_bd({{.*}}pkt_id = [[C:[0-9]+]]>
// CHECK-COUNT-3: aie.packet_flow(
// CHECK-NOT: aie.packet_flow(

air.channel @toA [1, 1] {channel_type = "npu_dma_packet"}
air.channel @toB [1, 1] {channel_type = "npu_dma_packet"}
air.channel @toC [1, 1] {channel_type = "npu_dma_packet"}
func.func @same_pattern_three_channels() {
  %c1 = arith.constant 1 : index
  air.launch (%la, %lb) in (%lc=%c1, %ld=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index

      // One buffer, one access pattern, three channels.
      air.herd @hp tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %t, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @toA[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @toB[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @toC[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dd0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }

      air.herd @hA tile (%x1, %y1) in (%s1=%c1_0, %r1=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @toA[] (%l[] [] []) : (memref<8xbf16, 2>)
        %dda = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hB tile (%x2, %y2) in (%s2=%c1_0, %r2=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @toB[] (%l[] [] []) : (memref<8xbf16, 2>)
        %ddb = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
      air.herd @hC tile (%x3, %y3) in (%s3=%c1_0, %r3=%c1_0)
            attributes {x_loc = 7 : i64, y_loc = 3 : i64} {
        %t, %l = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @toC[] (%l[] [] []) : (memref<8xbf16, 2>)
        %ddc = air.execute {memref.dealloc %l : memref<8xbf16, 2>}
      }
    }
  }
  return
}
