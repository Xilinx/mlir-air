//===- chain_lock_stage_once_per_fill.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 use-lock-race-condition-fix-v2=true" | FileCheck %s

// One fill of an L2 buffer, read twice through @ra and once through @rb. A
// chain lock passes the buffer from stage to stage once per fill, so @ra's
// second read would wait for a turn that never comes. The buffer keeps the
// counted lock: the fill waits for all three reads.

// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM, 0
// CHECK-NEXT: ^bb{{[0-9]+}}:
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c3_i32)
// CHECK: aie.use_lock(%{{.*}}, Release, %c3_i32)

module {
  air.channel @in [1, 1]
  air.channel @ra [1, 1]
  air.channel @rb [1, 1]
  func.func @twice(%arg0: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%a=%arg0) : memref<64xi32> {
      air.channel.put @in[] (%a[] [] []) : (memref<64xi32>)
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %l2 = memref.alloc() : memref<64xi32, 1 : i32>
        air.channel.get @in[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        air.channel.put @ra[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        air.channel.put @ra[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        air.channel.put @rb[] (%l2[] [] []) : (memref<64xi32, 1 : i32>)
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c2_0) {
          %buf = memref.alloc() : memref<64xi32, 2 : i32>
          %c0_h = arith.constant 0 : index
          %is0 = arith.cmpi eq, %y, %c0_h : index
          scf.if %is0 {
            air.channel.get @ra[%x, %y] (%buf[] [] []) : (memref<64xi32, 2 : i32>)
            air.channel.get @ra[%x, %y] (%buf[] [] []) : (memref<64xi32, 2 : i32>)
          } else {
            air.channel.get @rb[%x, %y] (%buf[] [] []) : (memref<64xi32, 2 : i32>)
          }
          memref.dealloc %buf : memref<64xi32, 2 : i32>
        }
        memref.dealloc %l2 : memref<64xi32, 1 : i32>
      }
    }
    return
  }
}
