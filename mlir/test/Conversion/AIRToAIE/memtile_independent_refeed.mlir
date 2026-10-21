// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// RUN: air-opt %s -air-to-aie="device=npu2 use-lock-race-condition-fix-v2=true" --split-input-file | FileCheck %s
// Independent resident buffers must drain on independent MM2S rings, even
// when they have the same replay count and share a logical packet channel.
// Later-stage data may depend on completing every replay of the first stage.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[A:[a-z0-9]+]],
// CHECK: ^[[A]]:
// CHECK: aie.dma_bd(
// CHECK: aie.next_bd ^[[A]]
// CHECK: aie.dma_start(MM2S, 1, ^[[B:[a-z0-9]+]],
// CHECK: ^[[B]]:
// CHECK: aie.dma_bd(
// CHECK: aie.next_bd ^[[B]]

air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @independent_replays(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.memtile_col = 0 : i32, air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.memtile_col = 0 : i32, air.refeed_count = 4 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %t2, %buf = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        air.channel.get @out[] (%buf[] [] []) : (memref<64xbf16, 2>)
        air.channel.put @done[] (%buf[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %buf : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}

// -----

// Ordinary packet buffers retain the existing shared-channel allocation.
// CHECK-LABEL: aie.device
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0,
// CHECK-NOT: aie.dma_start(MM2S, 1,
// CHECK: aie.shim_dma_allocation
air.channel @fill0 [1]
air.channel @fill1 [1]
air.channel @done [1]
air.channel @out [1] {channel_type = "npu_dma_packet"}
func.func @ordinary_packets_share(%a: memref<64xbf16>, %b: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b, %yy=%y) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> {
    air.channel.put @fill0[] (%aa[] [] []) : (memref<64xbf16>)
    air.channel.put @fill1[] (%bb[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %t0, %x = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.memtile_col = 0 : i32, air.refeed_count = 1 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      %t1, %z = air.execute -> (memref<64xbf16, 1>) {
        %allocated = memref.alloc() {air.memtile_col = 0 : i32, air.refeed_count = 1 : i32} : memref<64xbf16, 1>
        air.execute_terminator %allocated : memref<64xbf16, 1>
      }
      air.channel.get @fill0[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%x[] [] []) : (memref<64xbf16, 1>)
      air.channel.get @fill1[] (%z[] [] []) : (memref<64xbf16, 1>)
      air.channel.put @out[] (%z[] [] []) : (memref<64xbf16, 1>)
      %d0 = air.execute { memref.dealloc %x : memref<64xbf16, 1> }
      %d1 = air.execute { memref.dealloc %z : memref<64xbf16, 1> }
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        %t2, %buf = air.execute -> (memref<64xbf16, 2>) {
          %allocated = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %allocated : memref<64xbf16, 2>
        }
        air.channel.get @out[] (%buf[] [] []) : (memref<64xbf16, 2>)
        air.channel.put @done[] (%buf[] [] []) : (memref<64xbf16, 2>)
        %d2 = air.execute { memref.dealloc %buf : memref<64xbf16, 2> }
      }
    }
    air.channel.get @done[] (%yy[] [] []) : (memref<64xbf16>)
  }
  return
}
