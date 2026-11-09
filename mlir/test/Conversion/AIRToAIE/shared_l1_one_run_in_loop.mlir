//===- shared_l1_one_run_in_loop.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// Both cores touch the shared buffer once per iteration, after taking a
// block from their own channel. Each takes the shared buffer's lock around
// that one access, not from the top of the loop body: the consumer holding it
// across its own get would leave the producer's lock taken while the consumer
// waits.

// CHECK: aie.device
// CHECK-DAG: %[[PROD_TILE:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[CONS_TILE:.*]] = aie.tile(0, 3)

// CHECK: aie.core(%[[CONS_TILE]])
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK-NEXT: aie.use_lock(%{{.*}}_cons_lock, AcquireGreaterEqual
// CHECK-NEXT: func.call @take
// CHECK-NEXT: aie.use_lock(%lock_{{.*}}, Release
// CHECK-NEXT: aie.use_lock(%{{.*}}_prod_lock, Release

// CHECK: aie.core(%[[PROD_TILE]])
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK-NEXT: aie.use_lock(%{{.*}}_prod_lock, AcquireGreaterEqual
// CHECK-NEXT: func.call @give
// CHECK-NEXT: aie.use_lock(%lock_{{.*}}, Release
// CHECK-NEXT: aie.use_lock(%{{.*}}_cons_lock, Release

module {
  air.channel @to_prod [1, 1]
  air.channel @to_cons [1, 1]
  func.func private @give(memref<8xbf16, 2>, memref<8xbf16, 2>) attributes {link_with = "k.o"}
  func.func private @take(memref<8xbf16, 2>, memref<8xbf16, 2>) attributes {link_with = "k.o"}
  func.func @one_run(%a: memref<32xbf16>, %b: memref<32xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%aa=%a, %bb=%b) : memref<32xbf16>, memref<32xbf16> {
      air.channel.put @to_prod[] (%aa[] [] []) : (memref<32xbf16>)
      air.channel.put @to_cons[] (%bb[] [] []) : (memref<32xbf16>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        %shared = memref.alloc() : memref<8xbf16, 2>
        air.herd @prod tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s)
            args(%buf=%shared) : memref<8xbf16, 2>
            attributes {link_with = "k.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %in = memref.alloc() : memref<8xbf16, 2>
          scf.for %i = %c0 to %c4 step %c1_h {
            air.channel.get @to_prod[%tx, %ty] (%in[] [] []) : (memref<8xbf16, 2>)
            func.call @give(%in, %buf) : (memref<8xbf16, 2>, memref<8xbf16, 2>) -> ()
          }
          memref.dealloc %in : memref<8xbf16, 2>
        }
        air.herd @cons tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s)
            args(%buf=%shared) : memref<8xbf16, 2>
            attributes {link_with = "k.o", x_loc = 0 : i64, y_loc = 3 : i64} {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %in = memref.alloc() : memref<8xbf16, 2>
          scf.for %i = %c0 to %c4 step %c1_h {
            air.channel.get @to_cons[%tx, %ty] (%in[] [] []) : (memref<8xbf16, 2>)
            func.call @take(%buf, %in) : (memref<8xbf16, 2>, memref<8xbf16, 2>) -> ()
          }
          memref.dealloc %in : memref<8xbf16, 2>
        }
      }
    }
    return
  }
}
