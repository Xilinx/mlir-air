//===- air_segment_unroll_scf_if.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Tests for scf.if handling during air-to-std lowering with segment unroll.

// RUN: air-opt -air-to-std --split-input-file %s | FileCheck %s

// Positive test: scf.if selecting between L2 channel ops should be folded
// after L2 channel ops are replaced with wait_all.

// CHECK-LABEL: func.func @test_fold_l2_channel_scf_if
// The scf.if on segment unroll index returning an !airrt.event should be
// folded away after L2 channel ops are replaced with wait_all.
// CHECK-NOT: scf.if {{.*}}!airrt.event
// CHECK: airrt.wait_all
// CHECK: return

air.channel @chan_a [1, 1]
air.channel @chan_b [1, 1]

func.func @test_fold_l2_channel_scf_if(%arg0: memref<64xi32>) {
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

// -----

// Negative test: scf.if with compute ops alongside L2 channel ops must be
// preserved. After L2 channels are replaced with wait_all, the arith.addi
// remains, so the branches are not all-wait_all and the scf.if survives.

// CHECK-LABEL: func.func @test_preserve_scf_if_with_compute
// CHECK: scf.if
// CHECK: return

air.channel @chan_c [1, 1]
air.channel @chan_d [1, 1]

func.func @test_preserve_scf_if_with_compute(%arg0: memref<64xi32>) {
  %0 = air.launch async () in () args(%in=%arg0) : memref<64xi32>
      attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %segment = air.segment @seg2 async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1)
        attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64,
                    y_loc = 2 : i64, y_size = 2 : i64} {
      %c0_seg = arith.constant 0 : index
      %c1_seg = arith.constant 1 : index
      %c42 = arith.constant 42 : index

      %async_token, %buf = air.execute -> (memref<32xi32, 1 : i32>) {
        %alloc = memref.alloc() : memref<32xi32, 1 : i32>
        air.execute_terminator %alloc : memref<32xi32, 1 : i32>
      }

      // scf.if with L2 channel ops AND compute ops in branches.
      %cond = arith.cmpi eq, %ux, %c0_seg : index
      %3 = scf.if %cond -> (!air.async.token) {
        %put = air.channel.put async [%async_token] @chan_c[%c0_seg, %c0_seg]
            (%buf[] [] []) {id = 1 : i32} : (memref<32xi32, 1 : i32>)
        // Compute op that differs between branches.
        %sum = arith.addi %c42, %c42 : index
        scf.yield %put : !air.async.token
      } else {
        %put = air.channel.put async [%async_token] @chan_d[%c0_seg, %c0_seg]
            (%buf[] [] []) {id = 2 : i32} : (memref<32xi32, 1 : i32>)
        %prod = arith.muli %c42, %c42 : index
        scf.yield %put : !air.async.token
      }

      %herd = air.herd @herd2 async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg)
          args(%hux=%ux) : index
          attributes {id = 3 : i32} {
        %c0_h = arith.constant 0 : index
        %async_token_h, %l1buf = air.execute -> (memref<32xi32, 2>) {
          %alloc = memref.alloc() : memref<32xi32, 2>
          air.execute_terminator %alloc : memref<32xi32, 2>
        }
        %hcond = arith.cmpi eq, %hux, %c0_h : index
        scf.if %hcond {
          %get = air.channel.get async [%async_token_h] @chan_c[%c0_h, %c0_h]
              (%l1buf[] [] []) {id = 3 : i32} : (memref<32xi32, 2>)
        } else {
          %get = air.channel.get async [%async_token_h] @chan_d[%c0_h, %c0_h]
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

// -----

// Negative test: scf.if without else block is not touched.

// CHECK-LABEL: func.func @test_no_else_scf_if_untouched
// CHECK: scf.if
// CHECK: return

air.channel @chan_e [1, 1]

func.func @test_no_else_scf_if_untouched(%arg0: memref<64xi32>) {
  %0 = air.launch async () in () args(%in=%arg0) : memref<64xi32>
      attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %segment = air.segment @seg3 async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1)
        attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64,
                    y_loc = 2 : i64, y_size = 2 : i64} {
      %c0_seg = arith.constant 0 : index
      %c1_seg = arith.constant 1 : index

      %async_token, %buf = air.execute -> (memref<32xi32, 1 : i32>) {
        %alloc = memref.alloc() : memref<32xi32, 1 : i32>
        air.execute_terminator %alloc : memref<32xi32, 1 : i32>
      }

      // scf.if without else block — should NOT be folded.
      %cond = arith.cmpi eq, %ux, %c0_seg : index
      scf.if %cond {
        %put = air.channel.put async [%async_token] @chan_e[%c0_seg, %c0_seg]
            (%buf[] [] []) {id = 1 : i32} : (memref<32xi32, 1 : i32>)
      }

      %herd = air.herd @herd3 async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg)
          attributes {id = 3 : i32} {
        %c0_h = arith.constant 0 : index
        %async_token_h, %l1buf = air.execute -> (memref<32xi32, 2>) {
          %alloc = memref.alloc() : memref<32xi32, 2>
          air.execute_terminator %alloc : memref<32xi32, 2>
        }
        %get = air.channel.get async [%async_token_h] @chan_e[%c0_h, %c0_h]
            (%l1buf[] [] []) {id = 3 : i32} : (memref<32xi32, 2>)
        %dealloc = air.execute [%async_token_h] {
          memref.dealloc %l1buf : memref<32xi32, 2>
        }
        air.herd_terminator
      }
      %async_token_d = air.execute {
        memref.dealloc %buf : memref<32xi32, 1 : i32>
      }
    }
  }
  return
}
