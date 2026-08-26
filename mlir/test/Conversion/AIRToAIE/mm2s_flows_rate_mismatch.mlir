//===- mm2s_flows_rate_mismatch.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// Two packet flows out of ONE tile may share an MM2S channel only if they need
// the same number of firings per step. simpleDmaChannelAlloc multiplexes them
// onto a chained BD ring that serves each of its BDs once per pass, so a flow
// re-fed N times per step gets one firing per pass next to a flow that goes out
// once: it acquires N credits, is served 1, and blocks forever. Because the
// ring is in-order the other flow cannot proceed either and the design
// deadlocks having written nothing.
//
// The rate is air.refeed_count and NOT the trip count or the op count:
// air-annotate-refeed collapses the re-feed loop before air-to-aie runs, so by
// this point both of those read 1 for a re-fed flow.
//
// Note this hazard needs only ONE producer, which is what distinguishes it from
// the independent-producers rule next to it in spreadCollapsedPacketChannels:
// that rule is about the ORDER arrivals interleave in and is silent about rate.

// @xRefeed is re-fed 4x per step, @stOut goes out once. They must NOT end up on
// the same MM2S channel.
// CHECK-LABEL: aie.device
// CHECK-DAG: aie.dma_start(MM2S, 0
// CHECK-DAG: aie.dma_start(MM2S, 1

air.channel @xRefeed [1, 1] {channel_type = "npu_dma_packet", air.refeed_count = 4 : i32}
air.channel @stOut   [1, 1] {channel_type = "npu_dma_packet"}
func.func @rate_mismatch_splits() {
  %c1 = arith.constant 1 : index
  air.launch (%la, %lb) in (%lc=%c1, %ld=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @hm tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %t0, %p0 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t1, %p1 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.put @xRefeed[] (%p0[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @stOut[] (%p1[] [] []) : (memref<8xbf16, 2>)
        air.execute {
          memref.dealloc %p0 : memref<8xbf16, 2>
        }
        air.execute {
          memref.dealloc %p1 : memref<8xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// Control: the SAME two flows at the SAME rate. Equal re-feed counts are in
// step -- one pass of the ring advances both by one step's worth -- so sharing
// is sound and must be preserved. Without this section the check above would
// also be satisfied by a pass that simply split every packet flow onto its own
// channel, which would spend both MM2S channels of every tile in the project.
// CHECK-LABEL: aie.device
// CHECK: aie.dma_start(MM2S, 0
// CHECK-NOT: aie.dma_start(MM2S, 1

air.channel @xBoth [1, 1] {channel_type = "npu_dma_packet", air.refeed_count = 4 : i32}
air.channel @yBoth [1, 1] {channel_type = "npu_dma_packet", air.refeed_count = 4 : i32}
func.func @equal_rates_share() {
  %c1 = arith.constant 1 : index
  air.launch (%la, %lb) in (%lc=%c1, %ld=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @hs tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %t0, %p0 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        %t1, %p1 = air.execute -> (memref<8xbf16, 2>) {
          %x = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %x : memref<8xbf16, 2>
        }
        air.channel.put @xBoth[] (%p0[] [] []) : (memref<8xbf16, 2>)
        air.channel.put @yBoth[] (%p1[] [] []) : (memref<8xbf16, 2>)
        air.execute {
          memref.dealloc %p0 : memref<8xbf16, 2>
        }
        air.execute {
          memref.dealloc %p1 : memref<8xbf16, 2>
        }
      }
    }
  }
  return
}
