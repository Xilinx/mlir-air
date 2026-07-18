//===- air_channel_index_switch_dedicate.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Arm-varying get channel dedication. Two get channels each span both arms of an
// scf.index_switch: @xin_v is read in an scf.for in one arm (arm-varying dynamic
// read count) while @yin_f is read once in each arm (fixed). When several get
// channels would multiplex onto the tile's few physical S2MM channels and one is
// arm-varying, the pass pins each get channel to its own physical channel via
// air.tile_dma_channel: the arm-varying get gets a dedicated channel (1), the
// fixed get channel 0.

// CHECK-DAG: air.channel @xin_v [1, 1] {air.tile_dma_channel = 1 : i32}
// CHECK-DAG: air.channel @yin_f [1, 1] {air.tile_dma_channel = 0 : i32}

air.channel @xin_v [1, 1]
air.channel @yin_f [1, 1]
func.func @index_switch_dedicate(%ex: memref<2048xbf16>, %ey: memref<2048xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ex, %f=%ey) : memref<2048xbf16>, memref<2048xbf16> {
    %px = air.channel.put async @xin_v[] (%e[] [] []) {id = 1 : i32} : (memref<2048xbf16>)
    %py = air.channel.put async @yin_f[] (%f[] [] []) {id = 2 : i32} : (memref<2048xbf16>)
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %2 = air.herd @rms async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%arm=%c1_i32) : i32 {
        %armi = arith.index_cast %arm : i32 to index
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c7 = arith.constant 7 : index
        scf.index_switch %armi
        case 0 {
          %tA, %bA = air.execute -> (memref<2048xbf16, 2>) {
            %a = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %a : memref<2048xbf16, 2>
          }
          %tYA, %bYA = air.execute -> (memref<2048xbf16, 2>) {
            %a = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %a : memref<2048xbf16, 2>
          }
          %gl = scf.for %iv = %c0 to %c7 step %c1i iter_args(%t = %tA) -> (!air.async.token) {
            %gi = air.channel.get async [%t] @xin_v[] (%bA[] [] []) : (memref<2048xbf16, 2>)
            scf.yield %gi : !air.async.token
          }
          %gy = air.channel.get async [%tYA] @yin_f[] (%bYA[] [] []) : (memref<2048xbf16, 2>)
          air.execute [%gl] { memref.dealloc %bA : memref<2048xbf16, 2> }
          air.execute [%gy] { memref.dealloc %bYA : memref<2048xbf16, 2> }
          scf.yield
        }
        default {
          %tB, %bB = air.execute -> (memref<2048xbf16, 2>) {
            %b = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %b : memref<2048xbf16, 2>
          }
          %tYB, %bYB = air.execute -> (memref<2048xbf16, 2>) {
            %b = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %b : memref<2048xbf16, 2>
          }
          %gx = air.channel.get async [%tB] @xin_v[] (%bB[] [] []) : (memref<2048xbf16, 2>)
          %gy2 = air.channel.get async [%tYB] @yin_f[] (%bYB[] [] []) : (memref<2048xbf16, 2>)
          air.execute [%gx] { memref.dealloc %bB : memref<2048xbf16, 2> }
          air.execute [%gy2] { memref.dealloc %bYB : memref<2048xbf16, 2> }
          scf.yield
        }
      }
    }
  }
  return
}
