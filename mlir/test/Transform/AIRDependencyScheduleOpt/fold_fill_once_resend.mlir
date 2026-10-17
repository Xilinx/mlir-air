//===- fold_fill_once_resend.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" -split-input-file | FileCheck %s

// A buffer filled once per launch iteration and re-sent twice (an A-stationary
// GEMM tile) through a put with a free BD dimension: the loop is not folded into a
// dropped stride-0 dimension, it unrolls so the memtile lock counts both sends.

// CHECK-LABEL: @fold_fill_once_resend
// CHECK:       air.channel.get async {{.*}} @a_fill_in
// CHECK-NOT:   scf.for
// CHECK-COUNT-2: air.channel.put async {{.*}} @a_fill_out
// CHECK-NOT:   air.channel.put async {{.*}} @a_fill_out

module {
  air.channel @a_fill_in [1]
  air.channel @a_fill_out [1]
  func.func @fold_fill_once_resend(%arg0: memref<32x256xbf16>) {
    %0 = air.launch async () in () args(%a = %arg0) : memref<32x256xbf16> {
      %p = air.channel.put async @a_fill_in[] (%a[] [] []) {id = 1 : i32} : (memref<32x256xbf16>)
      %1 = air.segment @seg_a async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %t0, %buf = air.execute -> (memref<1x1x32x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x32x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x32x256xbf16, 1 : i32>
        }
        %g = air.channel.get async [%t0] @a_fill_in[] (%buf[] [] []) {id = 2 : i32} : (memref<1x1x32x256xbf16, 1 : i32>)
        %t = scf.for %iv = %c0 to %c2 step %c1 iter_args(%arg = %g) -> (!air.async.token) {
          %q = air.channel.put async [%arg] @a_fill_out[] (%buf[%c0, %c0, %c0] [%c32, %c32, %c8] [%c8, %c256, %c1]) {id = 3 : i32} : (memref<1x1x32x256xbf16, 1 : i32>)
          scf.yield %q : !air.async.token
        }
        %t1 = air.execute [%t] {
          memref.dealloc %buf : memref<1x1x32x256xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// The same loop when the producer refills the buffer every iteration: one fill
// per send, so the loop still folds into a single put.

// CHECK-LABEL: @fold_refill_per_iter
// CHECK:       air.channel.get async {{.*}} @a_refill_in
// CHECK-NOT:   scf.for
// CHECK:       air.channel.put async {{.*}} @a_refill_out
// CHECK-NOT:   air.channel.put async {{.*}} @a_refill_out

module {
  air.channel @a_refill_in [1]
  air.channel @a_refill_out [1]
  func.func @fold_refill_per_iter(%arg0: memref<32x256xbf16>) {
    %0 = air.launch async () in () args(%a = %arg0) : memref<32x256xbf16> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %w = air.wait_all async
      %pl = scf.for %iv = %c0 to %c2 step %c1 iter_args(%arg = %w) -> (!air.async.token) {
        %p = air.channel.put async [%arg] @a_refill_in[] (%a[] [] []) {id = 1 : i32} : (memref<32x256xbf16>)
        scf.yield %p : !air.async.token
      }
      %1 = air.segment @seg_a_refill async {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %t0, %buf = air.execute -> (memref<1x1x32x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x32x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x32x256xbf16, 1 : i32>
        }
        %g = air.channel.get async [%t0] @a_refill_in[] (%buf[] [] []) {id = 2 : i32} : (memref<1x1x32x256xbf16, 1 : i32>)
        %t = scf.for %iv = %c0_s to %c2_s step %c1_s iter_args(%arg = %g) -> (!air.async.token) {
          %q = air.channel.put async [%arg] @a_refill_out[] (%buf[%c0_s, %c0_s, %c0_s] [%c32, %c32, %c8] [%c8, %c256, %c1_s]) {id = 3 : i32} : (memref<1x1x32x256xbf16, 1 : i32>)
          scf.yield %q : !air.async.token
        }
        %t1 = air.execute [%t] {
          memref.dealloc %buf : memref<1x1x32x256xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
