// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" | FileCheck %s
//
// A packet endpoint has L3, L2, and L1 producers. Allocation must retain all
// three producers even when the last visited endpoint is not in L3, and the
// host transfer must receive shim metadata. All sources converge on one
// receiver with the same packet ID.
// CHECK: %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>
// CHECK: aie.packet_flow([[PID:[0-9]+]]) {
// CHECK-NEXT: aie.packet_source<%[[SHIM]], DMA : [[SCH:[0-9]+]]>
// CHECK-NEXT: aie.packet_dest<%[[DST:.*]], DMA : [[DCH:[0-9]+]]>
// CHECK: aie.packet_flow([[PID]]) {
// CHECK-NEXT: aie.packet_source<%logical_mem{{[^,]*}}, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST]], DMA : [[DCH]]>
// CHECK: aie.packet_flow([[PID]]) {
// CHECK-NEXT: aie.packet_source<%tile_3_3, DMA : {{[0-9]+}}>
// CHECK-NEXT: aie.packet_dest<%[[DST]], DMA : [[DCH]]>
// CHECK: aie.shim_dma_allocation @air_x(%[[SHIM]], MM2S, [[SCH]])
// CHECK: air.channel.put {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"

air.channel @tomt [1, 1]
air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
air.channel @r0 [1, 1]
func.func @mixed_producers(%host: memref<8xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%h=%host) : memref<8xbf16> {
    air.channel.put @x[] (%h[] [] []) {id = 100 : i32} : (memref<8xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      // L2 memtile producer: relay a core-produced buffer onto @x (MM2S from L2).
      %tm, %l2p = air.execute -> (memref<8xbf16, 1>) {
        %a2 = memref.alloc() {air.memtile_col = 1 : i32} : memref<8xbf16, 1>
        air.execute_terminator %a2 : memref<8xbf16, 1>
      }
      air.channel.get @tomt[] (%l2p[] [] []) : (memref<8xbf16, 1>)
      air.channel.put @x[] (%l2p[] [] []) : (memref<8xbf16, 1>)
      // Destination memtile buffer assembled from @x (2 producers: L2 + L1).
      %t, %l2 = air.execute -> (memref<3x8xbf16, 1>) {
        %alloc = memref.alloc() {air.no_split, air.memtile_col = 2 : i32} : memref<3x8xbf16, 1>
        air.execute_terminator %alloc : memref<3x8xbf16, 1>
      }
      air.channel.get @x[] (%l2[%c0,   %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c1_0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.get @x[] (%l2[%c2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<3x8xbf16, 1>)
      air.channel.put @r0[] (%l2[] [] []) : (memref<3x8xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<3x8xbf16, 1> }
      %dp = air.execute { memref.dealloc %l2p : memref<8xbf16, 1> }
      // L1 core producer #1: feeds the L2 relay via @tomt.
      air.herd @hp tile (%txp, %typ) in (%sxp=%c1_0, %syp=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @tomt[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dpp = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      // L1 core producer #2: puts directly onto @x (the mixed 2nd producer).
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @x[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<24xbf16, 2>) {
          %aa = memref.alloc() : memref<24xbf16, 2>
          air.execute_terminator %aa : memref<24xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<24xbf16, 2>)
        %dr = air.execute {memref.dealloc %l1 : memref<24xbf16, 2>}
      }
    }
  }
  return
}
