//===- segment_integer_args.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802 test-patterns=to-aie-mlir,after-clone-memcpys" | FileCheck %s

// The memtile transfers repeat a number of times computed from an i32 segment
// argument. The device has no value for a run-time scalar, so the copy of the
// loop in the device reads poison instead of the segment's argument; such a
// loop is lowered to a circular BD chain, which needs no count.
// CHECK: aie.device
// CHECK: %[[P:.*]] = ub.poison : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[P]] step %{{.*}} : i32

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
func.func @runtime_repeat_i32_loop(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c0_i = arith.constant 0 : i32
      %c1_i = arith.constant 1 : i32
      %c2_i = arith.constant 2 : i32
      %q = arith.divsi %sn, %c2_i : i32
      %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %i = %c0_i to %q step %c1_i : i32 {
        air.channel.get @w0[] (%l2[] [] []) : (memref<8xbf16, 1>)
        air.channel.put @r0[] (%l2[] [] []) : (memref<8xbf16, 1>)
      }
      %d_ = air.execute {memref.dealloc %l2 : memref<8xbf16, 1>}
      air.herd @h0 tile (%tx0, %ty0) in (%sx0=%c1_0, %sy0=%c1_0) attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %hn = arith.constant 4 : index
        %tok0, %l1a = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %hn step %c1_h {
          air.channel.put @w0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
          air.channel.get @r0[] (%l1a[] [] []) : (memref<8xbf16, 2>)
        }
        %d0 = air.execute {memref.dealloc %l1a : memref<8xbf16, 2>}
      }
    }
  }
  return
}
