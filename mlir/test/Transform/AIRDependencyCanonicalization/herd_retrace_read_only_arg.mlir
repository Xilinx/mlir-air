//===- herd_retrace_read_only_arg.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dependency-canonicalize | FileCheck %s

// The retrace of a herd that depends on another herd uses what the herd's body
// does with each arg. @r writes nothing: it reads %buf1 (written by @w) and
// %buf2 (read by the put before it), and holds the get matching that put. It
// must wait for @w but not for the put: both ends are async, so the put
// finishes only once the get inside @r has started.

// CHECK-LABEL: func.func @herd_retrace_two_buffers
// CHECK: air.execute
// CHECK: %[[BUF2:.*]], %{{.*}} = air.execute
// CHECK: %[[W:.*]] = air.herd @w async
// CHECK: air.channel.put async [%[[BUF2]]]
// The dependency list is exactly the alloc of %buf2 and @w.
// CHECK: air.herd @r async [%[[BUF2]], %[[W]]]

module {
  air.channel @feed [1, 1]
  func.func @herd_retrace_two_buffers() {
    air.segment @s {
      %c1_0 = arith.constant 1 : index
      %buf1 = memref.alloc() : memref<64xi32, 2>
      %buf2 = memref.alloc() : memref<64xi32, 2>
      air.herd @w tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%buf1) : memref<64xi32, 2> {
        %c0 = arith.constant 0 : index
        %v = arith.constant 7 : i32
        memref.store %v, %a[%c0] : memref<64xi32, 2>
      }
      air.channel.put @feed[] (%buf2[] [] []) : (memref<64xi32, 2>)
      air.herd @r tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%buf1, %b=%buf2) : memref<64xi32, 2>, memref<64xi32, 2> {
        %c0 = arith.constant 0 : index
        %x = memref.load %a[%c0] : memref<64xi32, 2>
        %y = memref.load %b[%c0] : memref<64xi32, 2>
        %l = memref.alloc() : memref<64xi32, 2>
        air.channel.get @feed[%tx, %ty] (%l[] [] []) : (memref<64xi32, 2>)
        memref.dealloc %l : memref<64xi32, 2>
      }
    }
    return
  }
}
