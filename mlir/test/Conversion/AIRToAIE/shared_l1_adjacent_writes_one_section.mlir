//===- shared_l1_adjacent_writes_one_section.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two BACK-TO-BACK writes to one shared L1 buffer take ONE lock section.
//
// The shared-L1 placer brackets each buffer-touching op on its own, which is
// deliberate: several buffers ping-ponging in one loop body each need their own
// acquire around their own access, and hoisting to the loop boundary would
// serialize ping against pong. But that rule, applied literally to two writes
// that are ADJACENT, releases the buffer to the DMA after the first one --
// before the second has happened -- and signals the consumer lock twice per
// production.
//
// That is not a corruption bug, it is a hang: the lock counts drift by one per
// production and the DMA and the core fall out of step. It cost a day to find
// once, on a compiler-emitted packet routing header sitting next to the payload
// write it belongs with. Writes separated by other work are untouched.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir' | FileCheck %s

// Anchor on the OWNER tile. @main is pinned at y_loc=2 and @helper at y_loc=3,
// but only the tile operand distinguishes their cores in the output -- and
// @helper's core opens with the same acquire-then-call pair. An unanchored
// `aie.core` would match whichever is emitted first and then fail at the
// vector.store, which reads exactly like the adjacency merge regressing.
// CHECK: %[[MAIN_TILE:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[CONS:.*]] = aie.lock(%[[MAIN_TILE]], {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD:.*]] = aie.lock(%[[MAIN_TILE]], {{.*}}) {init = 2 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}

// The owner core writes the payload and then the header word. ONE acquire in
// front of the pair, ONE release after it -- no lock op in between.
//
// The property is the LOCK SECTION, not line adjacency, so the writes are not
// required to print on immediately successive lines: loop plumbing or a
// scheduling change could legitimately land something harmless between them.
// A CHECK-NOT in each gap says exactly what matters -- no lock op anywhere
// from the acquire to the release -- and nothing more.
// CHECK: aie.core(%[[MAIN_TILE]])
// CHECK: aie.use_lock(%[[PROD]], AcquireGreaterEqual, %{{.*}})
// CHECK-NOT: aie.use_lock
// CHECK: func.call @zero_vectorized_bf16
// CHECK-NOT: aie.use_lock
// CHECK: vector.store
// CHECK-NOT: aie.use_lock
// CHECK: aie.use_lock(%[[CONS]], Release, %{{.*}})

module {
  func.func private @zero_vectorized_bf16(memref<8xbf16, 2 : i32>) attributes {link_with = "mv_int4_q4nx_bf16_v21.o", llvm.emit_c_interface}
  air.channel @out_chan []
  func.func @adjacent_writes(%arg0: memref<128xbf16>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1_0) args(%arg5=%arg0) : memref<128xbf16> {
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c8_1 = arith.constant 8 : index
      %c1_2 = arith.constant 1 : index
      air.channel.get  @out_chan[] (%arg5[] [%c16, %c8] [%c8_1, %c1_2]) : (memref<128xbf16>)
      air.segment @seg  {
        %alloc = memref.alloc() : memref<8xbf16, 2 : i32>
        %c1_3 = arith.constant 1 : index
        %c1_4 = arith.constant 1 : index
        air.herd @helper  tile (%arg6, %arg7) in (%arg8=%c1_3, %arg9=%c1_4) args(%arg10=%alloc) : memref<8xbf16, 2 : i32> attributes {link_with = "mv_int4_q4nx_bf16_v21.o", x_loc = 0 : i64, y_loc = 3 : i64} {
          %c0 = arith.constant 0 : index
          %c1_7 = arith.constant 1 : index
          %c16_8 = arith.constant 16 : index
          scf.for %arg11 = %c0 to %c16_8 step %c1_7 {
            func.call @zero_vectorized_bf16(%arg10) : (memref<8xbf16, 2 : i32>) -> ()
          }
        }
        %c1_5 = arith.constant 1 : index
        %c1_6 = arith.constant 1 : index
        air.herd @main  tile (%arg6, %arg7) in (%arg8=%c1_5, %arg9=%c1_6) args(%arg10=%alloc) : memref<8xbf16, 2 : i32> attributes {link_with = "mv_int4_q4nx_bf16_v21.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1_7 = arith.constant 1 : index
          %c16_8 = arith.constant 16 : index
          %hdr = arith.constant dense<0.0> : vector<2xbf16>
          scf.for %arg11 = %c0 to %c16_8 step %c1_7 {
            func.call @zero_vectorized_bf16(%arg10) : (memref<8xbf16, 2 : i32>) -> ()
            vector.store %hdr, %arg10[%c0] {alignment = 4 : i64} : memref<8xbf16, 2 : i32>, vector<2xbf16>
            air.channel.put  @out_chan[] (%arg10[] [] []) : (memref<8xbf16, 2 : i32>)
          }
        }
      }
    }
    return
  }
}
