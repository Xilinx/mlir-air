//===- air_split_l2_memref_loop_local_alloc.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// An L2 buffer allocated inside the loop that streams it (the weight-feed shape:
// one alloc per trip, filled by a get, drained by one put per destination tile).
// Splitting it re-traces the loop's dependencies, which inserts a wait_all for
// the iter_arg init OUTSIDE the loop; the sub-allocs live inside. The re-trace
// must not hand that wait_all a token defined in the loop body, or the pass
// emits `operand #0 does not dominate this use` and fails.

// The pass failing the verifier produces no output at all, so CHECK-LABEL
// alone guards the dominance bug. The rest asserts it did the work: both
// sub-allocs stay inside the loop, one per destination tile, and the 5120
// buffer is gone.
// CHECK-LABEL: func.func @loop_local_alloc_splits
// CHECK: scf.for
// CHECK: memref.alloc() : memref<2560xbf16, 1 : i32>
// CHECK: memref.alloc() : memref<2560xbf16, 1 : i32>
// CHECK-NOT: memref<5120xbf16, 1 : i32>

air.channel @inW [1]
air.channel @wL2ToL1 [1, 2]
func.func @loop_local_alloc_splits(%arg0: memref<5120xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg1) in (%arg2=%c1) args(%arg3=%arg0) : memref<5120xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5120 = arith.constant 5120 : index
    %1 = scf.for %arg4 = %c0 to %c4 step %c1_0 iter_args(%arg5 = %c0) -> (index) {
      scf.yield %arg5 : index
    }
    %2 = air.segment @segment_0 async {
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4_3 = arith.constant 4 : index
      %3 = air.wait_all async
      %4 = scf.for %arg4 = %c0_1 to %c4_3 step %c1_2 iter_args(%arg5 = %3) -> (!air.async.token) {
        %async_token, %results = air.execute -> (memref<5120xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<5120xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<5120xbf16, 1 : i32>
        }
        %6 = air.channel.get async [%arg5, %async_token]  @inW[%c0_1] (%results[] [] []) {id = 1 : i32} : (memref<5120xbf16, 1 : i32>)
        %7 = air.channel.put async [%6]  @wL2ToL1[%c0_1, %c0_1] (%results[0] [2560] [1]) {id = 2 : i32} : (memref<5120xbf16, 1 : i32>)
        %8 = air.channel.put async [%6]  @wL2ToL1[%c0_1, %c1_2] (%results[2560] [2560] [1]) {id = 3 : i32} : (memref<5120xbf16, 1 : i32>)
        %async_token_4 = air.execute [%8, %7] {
          memref.dealloc %results : memref<5120xbf16, 1 : i32>
        }
        %9 = air.wait_all async [%7, %8]
        scf.yield %9 : !air.async.token
      }
      %5 = air.herd @herd_0 async tile (%arg4, %arg5) in (%arg6=%c1_2, %arg7=%c2) attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c0_5 = arith.constant 0 : index
        %c1_6 = arith.constant 1 : index
        %c4_7 = arith.constant 4 : index
        %6 = scf.for %arg8 = %c0_5 to %c4_7 step %c1_6 iter_args(%arg9 = %c0_5) -> (index) {
          %async_token, %results = air.execute -> (memref<2560xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<2560xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<2560xbf16, 2 : i32>
          }
          %7 = air.channel.get async [%async_token]  @wL2ToL1[%arg4, %arg5] (%results[] [] []) {id = 4 : i32} : (memref<2560xbf16, 2 : i32>)
          %async_token_8 = air.execute [%7] {
            memref.dealloc %results : memref<2560xbf16, 2 : i32>
          }
          scf.yield %arg9 : index
        }
      }
    }
  }
  return
}
