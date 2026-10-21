//===- air_channel_index_switch_memtile.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802 test-patterns=to-aie-mlir-after-specialize-channel" | FileCheck %s --implicit-check-not=memref.copy

// L2 arm buffers use the memtile allocator. A compute-tile channel pin on
// their shared channel declaration would incorrectly constrain the producer
// core (and number unrelated memtiles as if they were one compute tile).
// The L1 dedication case is covered by air_channel_index_switch_dedicate.mlir.
// CHECK: aie.device
// CHECK-DAG: air.channel @xin_v [1, 1]{{$}}
// CHECK-DAG: air.channel @yin_f [1, 1]{{$}}
// CHECK-DAG: memref.alloc() {air.memtile_col = 2 : i32}

air.channel @xin_v [1, 1]
air.channel @yin_f [1, 1]
air.channel @relay [1, 1]
func.func @index_switch_memtile(%ex: memref<2048xbf16>, %ey: memref<2048xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ex, %f=%ey) : memref<2048xbf16>, memref<2048xbf16> {
    %px = air.channel.put async @xin_v[] (%e[] [] []) {id = 1 : i32} : (memref<2048xbf16>)
    %py = air.channel.put async @yin_f[] (%f[] [] []) {id = 2 : i32} : (memref<2048xbf16>)
    %1 = air.segment async {
      %arm = arith.constant 1 : i32
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @dummy async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {}
        %armi = arith.index_cast %arm : i32 to index
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c7 = arith.constant 7 : index
        scf.index_switch %armi
        case 0 {
          %tA, %bA = air.execute -> (memref<2048xbf16, 1>) {
            %a = memref.alloc() {air.memtile_col = 2 : i32} : memref<2048xbf16, 1>
            air.execute_terminator %a : memref<2048xbf16, 1>
          }
          %tYA, %bYA = air.execute -> (memref<2048xbf16, 1>) {
            %a = memref.alloc() {air.memtile_col = 2 : i32} : memref<2048xbf16, 1>
            air.execute_terminator %a : memref<2048xbf16, 1>
          }
          %gl = scf.for %iv = %c0 to %c7 step %c1i iter_args(%t = %tA) -> (!air.async.token) {
            %gi = air.channel.get async [%t] @xin_v[] (%bA[] [] []) : (memref<2048xbf16, 1>)
            scf.yield %gi : !air.async.token
          }
          %gy = air.channel.get async [%tYA] @yin_f[] (%bYA[] [] []) : (memref<2048xbf16, 1>)
          air.channel.put @relay[] (%bA[] [] []) : (memref<2048xbf16, 1>)
          air.execute [%gl] { memref.dealloc %bA : memref<2048xbf16, 1> }
          air.execute [%gy] { memref.dealloc %bYA : memref<2048xbf16, 1> }
          scf.yield
        }
        default {
          %tB, %bB = air.execute -> (memref<2048xbf16, 1>) {
            %b = memref.alloc() {air.memtile_col = 2 : i32} : memref<2048xbf16, 1>
            air.execute_terminator %b : memref<2048xbf16, 1>
          }
          %tYB, %bYB = air.execute -> (memref<2048xbf16, 1>) {
            %b = memref.alloc() {air.memtile_col = 2 : i32} : memref<2048xbf16, 1>
            air.execute_terminator %b : memref<2048xbf16, 1>
          }
          %gx = air.channel.get async [%tB] @xin_v[] (%bB[] [] []) : (memref<2048xbf16, 1>)
          %gy2 = air.channel.get async [%tYB] @yin_f[] (%bYB[] [] []) : (memref<2048xbf16, 1>)
          air.channel.put @relay[] (%bB[] [] []) : (memref<2048xbf16, 1>)
          air.execute [%gx] { memref.dealloc %bB : memref<2048xbf16, 1> }
          air.execute [%gy2] { memref.dealloc %bYB : memref<2048xbf16, 1> }
          scf.yield
        }
    }
  }
  return
}
