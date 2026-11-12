//===- chain_lock_stage_loops.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 use-lock-race-condition-fix-v2=true" | FileCheck %s

// Two fills of an L2 buffer, and two readers (@ra, @rb) each in a loop of two
// trips: each stage moves the buffer once per fill, counted by the trips of
// the loops not around all its transfers (not by its op sites, one each), so
// the buffer takes the chain lock: a ping-pong pair and three locks.
// (chain_lock_stage_once_per_fill.mlir: a stage moving it twice per fill.)

// CHECK-DAG: aie.lock(%[[MT:.*]], 2) {init = 0 : i32}
// CHECK-DAG: aie.lock(%[[MT]], 1) {init = 0 : i32}
// CHECK-DAG: aie.lock(%[[MT]], 0) {init = 2 : i32}
// CHECK: aie.memtile_dma

module {
  air.channel @in [1, 1]
  air.channel @ra [1, 1]
  air.channel @rb [1, 1]
  func.func @looped(%arg0: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%a=%arg0) : memref<64xi32> {
      air.channel.put @in[] (%a[] [] []) : (memref<64xi32>)
      air.channel.put @in[] (%a[] [] []) : (memref<64xi32>)
      air.segment @seg {
        %c0_0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %l2 = memref.alloc() : memref<64xi32, 1 : i32>
        air.channel.get @in[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        air.channel.get @in[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        scf.for %i = %c0_0 to %c2_0 step %c1_0 {
          air.channel.put @ra[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        }
        scf.for %i = %c0_0 to %c2_0 step %c1_0 {
          air.channel.put @rb[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c2_0) {
          %buf = memref.alloc() : memref<64xi32, 2 : i32>
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %is0 = arith.cmpi eq, %y, %c0_h : index
          scf.if %is0 {
            scf.for %i = %c0_h to %c2_h step %c1_h {
              air.channel.get @ra[%x, %y] (%buf[] [] []) : (memref<64xi32, 2 : i32>)
            }
          } else {
            scf.for %i = %c0_h to %c2_h step %c1_h {
              air.channel.get @rb[%x, %y] (%buf[] [] []) : (memref<64xi32, 2 : i32>)
            }
          }
          memref.dealloc %buf : memref<64xi32, 2 : i32>
        }
        memref.dealloc %l2 : memref<64xi32, 1 : i32>
      }
    }
    return
  }
}
