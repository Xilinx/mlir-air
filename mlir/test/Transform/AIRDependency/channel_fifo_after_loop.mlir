//===- channel_fifo_after_loop.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -split-input-file | FileCheck %s

// Puts of unrelated buffers on one channel before, inside and after a loop: the
// put after the loop is ordered after the loop's, through its token result, and
// after the put before the loop, which the loop's token does not wait on.

// CHECK-LABEL: func.func @after_loop
// CHECK: %[[P:.*]] = air.channel.put async{{.*}}@c[] {{.*}} {id = 1 : i32}
// CHECK: %[[L:.*]] = scf.for
// CHECK: air.channel.put async{{.*}}@c[] {{.*}} {id = 2 : i32}
// CHECK: air.channel.put async [%[[P]], %[[L]]]  @c[] {{.*}} {id = 3 : i32}

module {
  air.channel @c [1]
  func.func @after_loop(%x: memref<64xi32>, %h: memref<4x16xi32>, %y: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%a=%x, %b=%h, %d=%y) : memref<64xi32>, memref<4x16xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      air.channel.put @c[] (%a[] [] []) {id = 1 : i32} : (memref<64xi32>)
      scf.for %iv = %c0 to %c4 step %c1_0 {
        air.channel.put @c[] (%b[%iv, %c0] [%c1_0, %c16] [%c16, %c1_0]) {id = 2 : i32} : (memref<4x16xi32>)
      }
      air.channel.put @c[] (%d[] [] []) {id = 3 : i32} : (memref<64xi32>)
    }
    return
  }
}

// -----

// The same after a loop nest: the put after it follows the outer loop.

// CHECK-LABEL: func.func @after_loop_nest
// CHECK: %[[P:.*]] = air.channel.put async{{.*}}@c[] {{.*}} {id = 1 : i32}
// CHECK: %[[L:.*]] = scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@c[] {{.*}} {id = 2 : i32}
// CHECK: air.channel.put async [%[[P]], %[[L]]]  @c[] {{.*}} {id = 3 : i32}

module {
  air.channel @c [1]
  func.func @after_loop_nest(%x: memref<64xi32>, %h: memref<4x16xi32>, %y: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%a=%x, %b=%h, %d=%y) : memref<64xi32>, memref<4x16xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      air.channel.put @c[] (%a[] [] []) {id = 1 : i32} : (memref<64xi32>)
      scf.for %o = %c0 to %c2 step %c1_0 {
        scf.for %iv = %c0 to %c2 step %c1_0 {
          air.channel.put @c[] (%b[%iv, %c0] [%c1_0, %c16] [%c16, %c1_0]) {id = 2 : i32} : (memref<4x16xi32>)
        }
      }
      air.channel.put @c[] (%d[] [] []) {id = 3 : i32} : (memref<64xi32>)
    }
    return
  }
}
