//===- air_lock_count_exclusive_arms.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 emit-herd-lock=true" | FileCheck %s

// An L2 buffer filled once and read by one of two puts, in the two arms of an
// scf.if on a dispatch's scalar (a class of waves, say): only one read follows
// each fill. The fill waits for one read, not two -- it would wait forever for
// the second.

// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[FULL:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK: aie.use_lock(%[[EMPTY:.*]], Release, %c1_i32)
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.use_lock(%[[FULL]], AcquireGreaterEqual, %c1_i32)
// CHECK: aie.use_lock(%[[EMPTY]], Release, %c1_i32)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[EMPTY]], AcquireGreaterEqual, %c1_i32)
// CHECK: aie.use_lock(%[[FULL]], Release, %c1_i32)

module {
  air.channel @in [1, 1]
  air.channel @toA [1, 1]
  air.channel @toB [1, 1]
  func.func @arms(%arg0: memref<64xi32>, %sel: i32) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%a=%arg0, %s=%sel) : memref<64xi32>, i32 {
      air.channel.put @in[] (%a[] [] []) : (memref<64xi32>)
      air.segment @seg args(%ss=%s) : i32 {
        %c1_0 = arith.constant 1 : index
        %c0_i32 = arith.constant 0 : i32
        %l2 = memref.alloc() : memref<64xi32, 1>
        air.channel.get @in[] (%l2[] [] []) : (memref<64xi32, 1>)
        %cond = arith.cmpi eq, %ss, %c0_i32 : i32
        scf.if %cond {
          air.channel.put @toA[] (%l2[] [] []) : (memref<64xi32, 1>)
        } else {
          air.channel.put @toB[] (%l2[] [] []) : (memref<64xi32, 1>)
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%v=%ss) : i32 {
          %buf = memref.alloc() : memref<64xi32, 2>
          %c0_h = arith.constant 0 : i32
          %cnd = arith.cmpi eq, %v, %c0_h : i32
          scf.if %cnd {
            air.channel.get @toA[%x, %y] (%buf[] [] []) : (memref<64xi32, 2>)
          } else {
            air.channel.get @toB[%x, %y] (%buf[] [] []) : (memref<64xi32, 2>)
          }
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<64xi32, 1>
      }
    }
    return
  }
}
