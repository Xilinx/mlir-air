//===- fuse_channels_exclusive_arms.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-fuse-channels="aggressive-mode=L1" | FileCheck %s

// Two classes of waves, chosen by a dispatch scalar, each moving the same two
// L2 buffers into the same two L1 buffers through channels of its own. Only
// one arm runs, and the arms are alike: their channels become one by renaming,
// and each arm keeps its own transfers (time multiplexing would move the
// other arm's into it).

// CHECK-LABEL: func.func @classes
// CHECK: scf.if
// CHECK-NEXT: air.channel.put {{.*}}@[[CH:[a-zA-Z0-9_]+]][
// CHECK-NEXT: air.channel.put {{.*}}@[[CH]][
// CHECK-NOT: air.channel.put
// CHECK: } else {
// CHECK-NEXT: air.channel.put {{.*}}@[[CH]][
// CHECK-NEXT: air.channel.put {{.*}}@[[CH]][
// CHECK: air.herd
// CHECK: scf.if
// CHECK-NEXT: air.channel.get {{.*}}@[[CH]][
// CHECK-NEXT: air.channel.get {{.*}}@[[CH]][
// CHECK-NOT: air.channel.get
// CHECK: } else {
// CHECK-NEXT: air.channel.get {{.*}}@[[CH]][
// CHECK-NEXT: air.channel.get {{.*}}@[[CH]][

module {
  air.channel @a0 [1, 1]
  air.channel @b0 [1, 1]
  air.channel @a1 [1, 1]
  air.channel @b1 [1, 1]
  func.func @classes(%sel: i32) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%s=%sel) : i32 {
      air.segment @seg args(%ss=%s) : i32 {
        %c1_0 = arith.constant 1 : index
        %c0_i32 = arith.constant 0 : i32
        %x = memref.alloc() : memref<64xi32, 1>
        %w = memref.alloc() : memref<64xi32, 1>
        %cond = arith.cmpi eq, %ss, %c0_i32 : i32
        scf.if %cond {
          air.channel.put @a0[] (%x[] [] []) : (memref<64xi32, 1>)
          air.channel.put @b0[] (%w[] [] []) : (memref<64xi32, 1>)
        } else {
          air.channel.put @a1[] (%x[] [] []) : (memref<64xi32, 1>)
          air.channel.put @b1[] (%w[] [] []) : (memref<64xi32, 1>)
        }
        air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%v=%ss) : i32 attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %bx = memref.alloc() : memref<64xi32, 2>
          %bw = memref.alloc() : memref<64xi32, 2>
          %c0_h = arith.constant 0 : i32
          %cnd = arith.cmpi eq, %v, %c0_h : i32
          scf.if %cnd {
            air.channel.get @a0[%tx, %ty] (%bx[] [] []) : (memref<64xi32, 2>)
            air.channel.get @b0[%tx, %ty] (%bw[] [] []) : (memref<64xi32, 2>)
          } else {
            air.channel.get @a1[%tx, %ty] (%bx[] [] []) : (memref<64xi32, 2>)
            air.channel.get @b1[%tx, %ty] (%bw[] [] []) : (memref<64xi32, 2>)
          }
          memref.dealloc %bx : memref<64xi32, 2>
          memref.dealloc %bw : memref<64xi32, 2>
        }
        memref.dealloc %x : memref<64xi32, 1>
        memref.dealloc %w : memref<64xi32, 1>
      }
    }
    return
  }
}
