//===- unroll_fill_once_resend.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" -split-input-file | FileCheck %s

// A buffer filled once per launch iteration and re-sent 16 times (a B-stationary
// GEMM tile): the redundant-repeat loop must unroll so the memtile lock counts the
// 16 re-sends. Left rolled, the lock is lowered as 1 and the memtile waits for 15
// refills that never come.

// CHECK-LABEL: @unroll_fill_once_resend
// CHECK:       air.channel.get async {{.*}} @chan_fill_in
// CHECK-NOT:   scf.for
// CHECK-COUNT-16: air.channel.put async {{.*}} @chan_fill_out
// CHECK-NOT:   air.channel.put async {{.*}} @chan_fill_out

module {
  air.channel @chan_fill_in [1]
  air.channel @chan_fill_out [1]
  func.func @unroll_fill_once_resend(%arg0: memref<768x96xbf16>) {
    %0 = air.launch async () in () args(%a = %arg0) : memref<768x96xbf16> {
      %p = air.channel.put async @chan_fill_in[] (%a[] [] []) {id = 1 : i32} : (memref<768x96xbf16>)
      %1 = air.segment @seg_fill async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c12 = arith.constant 12 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c96 = arith.constant 96 : index
        %c6144 = arith.constant 6144 : index
        %t0, %buf = air.execute -> (memref<1x1x768x96xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x768x96xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x768x96xbf16, 1 : i32>
        }
        %g = air.channel.get async [%t0] @chan_fill_in[] (%buf[] [] []) {id = 2 : i32} : (memref<1x1x768x96xbf16, 1 : i32>)
        %t = scf.for %iv = %c0 to %c16 step %c1 iter_args(%arg = %g) -> (!air.async.token) {
          %q = air.channel.put async [%arg] @chan_fill_out[] (%buf[%c0, %c0, %c0, %c0] [%c12, %c12, %c64, %c8] [%c6144, %c8, %c96, %c1]) {id = 3 : i32} : (memref<1x1x768x96xbf16, 1 : i32>)
          scf.yield %q : !air.async.token
        }
        %t1 = air.execute [%t] {
          memref.dealloc %buf : memref<1x1x768x96xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// The same loop when the producer refills the buffer every iteration: one fill
// per send, so the loop stays rolled (the case the redundant-unroll guard is for).

// CHECK-LABEL: @refill_per_iter_preserved
// CHECK:       scf.for
// CHECK:         air.channel.put async {{.*}} @chan_refill_out
// CHECK-NOT:     air.channel.put async {{.*}} @chan_refill_out
// CHECK:         scf.yield

module {
  air.channel @chan_refill_in [1]
  air.channel @chan_refill_out [1]
  func.func @refill_per_iter_preserved(%arg0: memref<768x96xbf16>) {
    %0 = air.launch async () in () args(%a = %arg0) : memref<768x96xbf16> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %w = air.wait_all async
      %pl = scf.for %iv = %c0 to %c16 step %c1 iter_args(%arg = %w) -> (!air.async.token) {
        %p = air.channel.put async [%arg] @chan_refill_in[] (%a[] [] []) {id = 1 : i32} : (memref<768x96xbf16>)
        scf.yield %p : !air.async.token
      }
      %1 = air.segment @seg_refill async {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c12 = arith.constant 12 : index
        %c16_s = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c96 = arith.constant 96 : index
        %c6144 = arith.constant 6144 : index
        %t0, %buf = air.execute -> (memref<1x1x768x96xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x768x96xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x768x96xbf16, 1 : i32>
        }
        %g = air.channel.get async [%t0] @chan_refill_in[] (%buf[] [] []) {id = 2 : i32} : (memref<1x1x768x96xbf16, 1 : i32>)
        %t = scf.for %iv = %c0_s to %c16_s step %c1_s iter_args(%arg = %g) -> (!air.async.token) {
          %q = air.channel.put async [%arg] @chan_refill_out[] (%buf[%c0_s, %c0_s, %c0_s, %c0_s] [%c12, %c12, %c64, %c8] [%c6144, %c8, %c96, %c1_s]) {id = 3 : i32} : (memref<1x1x768x96xbf16, 1 : i32>)
          scf.yield %q : !air.async.token
        }
        %t1 = air.execute [%t] {
          memref.dealloc %buf : memref<1x1x768x96xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
