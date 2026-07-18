//===- air_channel_index_switch_relay_break.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Get->put relay break inside an scf.index_switch. Each arm both FILLS a buffer
// via air.channel.get (S2MM) and FORWARDS it via air.channel.put (MM2S) using
// the SAME memref. Unifying the get and put onto one buffer would create a DMA
// relay whose S2MM and MM2S halves hold distinct lock pairs and deadlock, so the
// pass copies the get's buffer into a fresh forward buffer for the put. The
// get-fill buffer and the put forward buffer are therefore DISTINCT: the tile
// ends up with TWO L1 buffers of the type (plus the herd RTP buffer), not one
// shared get+put buffer.

// CHECK: aie.device
// CHECK-COUNT-2: aie.buffer(%{{.*}}) {{.*}} : memref<2048xbf16, 2
// CHECK-NOT: aie.buffer(%{{.*}}) {{.*}} : memref<2048xbf16, 2

air.channel @xin2 [1, 1]
air.channel @xout2 [1, 1]
func.func @index_switch_relay_break(%ext: memref<2048xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<2048xbf16> {
    %p = air.channel.put async @xin2[] (%e[] [] []) {id = 1 : i32} : (memref<2048xbf16>)
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %2 = air.herd @rms async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%arm=%c1_i32) : i32 {
        %armi = arith.index_cast %arm : i32 to index
        scf.index_switch %armi
        case 0 {
          %tA, %bA = air.execute -> (memref<2048xbf16, 2>) {
            %a = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %a : memref<2048xbf16, 2>
          }
          %gi = air.channel.get async [%tA] @xin2[] (%bA[] [] []) : (memref<2048xbf16, 2>)
          %po = air.channel.put async [%gi] @xout2[] (%bA[] [] []) : (memref<2048xbf16, 2>)
          air.execute [%po] { memref.dealloc %bA : memref<2048xbf16, 2> }
          scf.yield
        }
        default {
          %tB, %bB = air.execute -> (memref<2048xbf16, 2>) {
            %b = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %b : memref<2048xbf16, 2>
          }
          %gi2 = air.channel.get async [%tB] @xin2[] (%bB[] [] []) : (memref<2048xbf16, 2>)
          %po2 = air.channel.put async [%gi2] @xout2[] (%bB[] [] []) : (memref<2048xbf16, 2>)
          air.execute [%po2] { memref.dealloc %bB : memref<2048xbf16, 2> }
          scf.yield
        }
      }
    }
  }
  return
}
