//===- air_segment_unroll_scf_if.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that scf.if ops on segment unroll indices wrapping L2 channel ops
// are correctly folded during air-to-std lowering. When air-to-aie resolves
// segment-level channel specialization into separate devices, the original
// segment body's scf.if branches contain different L2 channel ops. After
// air-to-std replaces L2 channel ops with wait_all, both branches become
// identical and the scf.if should be folded away to avoid airrt-to-npu
// legalization failures on !airrt.event types.

// RUN: air-opt -air-to-std %s | FileCheck %s

// CHECK-LABEL: func.func @test
// The scf.if on segment unroll index should be folded away after
// L2 channel ops are replaced with wait_all.
// CHECK-NOT: scf.if
// CHECK: return

air.channel @chan_a [1, 1]
air.channel @chan_b [1, 1]

func.func @test(%arg0: memref<64xi32>) {
  %0 = air.launch async () in () args(%in=%arg0) : memref<64xi32>
      attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %segment = air.segment @seg async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1)
        attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64,
                    y_loc = 2 : i64, y_size = 2 : i64} {
      %c0_seg = arith.constant 0 : index
      %c1_seg = arith.constant 1 : index

      %async_token, %buf = air.execute -> (memref<32xi32, 1 : i32>) {
        %alloc = memref.alloc() : memref<32xi32, 1 : i32>
        air.execute_terminator %alloc : memref<32xi32, 1 : i32>
      }

      // scf.if on segment unroll index selecting different L2 channels.
      %cond = arith.cmpi eq, %ux, %c0_seg : index
      %3 = scf.if %cond -> (!air.async.token) {
        %put = air.channel.put async [%async_token] @chan_a[%c0_seg, %c0_seg]
            (%buf[] [] []) {id = 1 : i32} : (memref<32xi32, 1 : i32>)
        scf.yield %put : !air.async.token
      } else {
        %put = air.channel.put async [%async_token] @chan_b[%c0_seg, %c0_seg]
            (%buf[] [] []) {id = 2 : i32} : (memref<32xi32, 1 : i32>)
        scf.yield %put : !air.async.token
      }

      // Herd with matching channel gets (required for channel lowering).
      %herd = air.herd @herd async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg)
          args(%hux=%ux) : index
          attributes {id = 3 : i32} {
        %c0_h = arith.constant 0 : index
        %async_token_h, %l1buf = air.execute -> (memref<32xi32, 2>) {
          %alloc = memref.alloc() : memref<32xi32, 2>
          air.execute_terminator %alloc : memref<32xi32, 2>
        }
        %hcond = arith.cmpi eq, %hux, %c0_h : index
        scf.if %hcond {
          %get = air.channel.get async [%async_token_h] @chan_a[%c0_h, %c0_h]
              (%l1buf[] [] []) {id = 3 : i32} : (memref<32xi32, 2>)
        } else {
          %get = air.channel.get async [%async_token_h] @chan_b[%c0_h, %c0_h]
              (%l1buf[] [] []) {id = 4 : i32} : (memref<32xi32, 2>)
        }
        %dealloc = air.execute [%async_token_h] {
          memref.dealloc %l1buf : memref<32xi32, 2>
        }
        air.herd_terminator
      }
      %async_token_d = air.execute [%3] {
        memref.dealloc %buf : memref<32xi32, 1 : i32>
      }
    }
  }
  return
}
