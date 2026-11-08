//===- shared_l1_one_run_nested.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// The producer fills the shared buffer in an inner loop, and the consumer
// reads it once per outer iteration. The producer's one access run is not in
// the outer loop's body, so it takes the buffer for the whole outer
// iteration, not once per inner iteration.

// CHECK: aie.device
// CHECK-DAG: %[[PROD_TILE:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[CONS_TILE:.*]] = aie.tile(0, 3)

// CHECK: aie.core(%[[PROD_TILE]])
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}_prod_lock, AcquireGreaterEqual
// CHECK: scf.for
// CHECK-NOT: {{_prod_lock|_cons_lock}}
// CHECK: func.call @give
// CHECK-NOT: {{_prod_lock|_cons_lock}}
// CHECK: }
// CHECK-NEXT: aie.use_lock(%{{.*}}_cons_lock, Release

module {
  air.channel @to_prod [1, 1]
  func.func private @give(memref<8xbf16, 2>, memref<8xbf16, 2>) attributes {link_with = "k.o"}
  func.func private @take(memref<8xbf16, 2>, memref<8xbf16, 2>) attributes {link_with = "k.o"}
  func.func @nested(%a: memref<64xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%aa=%a) : memref<64xbf16> {
      air.channel.put @to_prod[] (%aa[] [] []) : (memref<64xbf16>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        %shared = memref.alloc() : memref<8xbf16, 2>
        air.herd @prod tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s)
            args(%buf=%shared) : memref<8xbf16, 2>
            attributes {link_with = "k.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %in = memref.alloc() : memref<8xbf16, 2>
          scf.for %i = %c0 to %c4 step %c1_h {
            scf.for %j = %c0 to %c2 step %c1_h {
              air.channel.get @to_prod[%tx, %ty] (%in[] [] []) : (memref<8xbf16, 2>)
              func.call @give(%in, %buf) : (memref<8xbf16, 2>, memref<8xbf16, 2>) -> ()
            }
          }
          memref.dealloc %in : memref<8xbf16, 2>
        }
        air.herd @cons tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s)
            args(%buf=%shared) : memref<8xbf16, 2>
            attributes {link_with = "k.o", x_loc = 0 : i64, y_loc = 3 : i64} {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %out = memref.alloc() : memref<8xbf16, 2>
          scf.for %i = %c0 to %c4 step %c1_h {
            func.call @take(%buf, %out) : (memref<8xbf16, 2>, memref<8xbf16, 2>) -> ()
          }
          memref.dealloc %out : memref<8xbf16, 2>
        }
      }
    }
    return
  }
}
