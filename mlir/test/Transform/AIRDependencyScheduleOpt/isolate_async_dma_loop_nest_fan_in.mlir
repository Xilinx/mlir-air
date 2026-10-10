//===- isolate_async_dma_loop_nest_fan_in.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests="scope=func" --split-input-file | FileCheck %s

// Two puts on different channels behind a chain of fan-in joins: %lN and %rN
// each wait on both ops of the level before, so there are 2^30 paths back to
// the loop's start. Neither put depends on the other, so each gets its own
// loop, and deciding that must not walk every path.

// CHECK-LABEL: func.func @fan_in
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_a
// CHECK-NOT: air.channel.put
// CHECK: scf.yield
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_b
// CHECK: scf.yield

air.channel @channel_a [1, 1]
air.channel @channel_b [1, 1]
func.func @fan_in(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t0 = air.wait_all async
  %t = scf.for %i = %c0 to %c8 step %c1 iter_args(%l0 = %t0) -> (!air.async.token) {
    %r0 = air.wait_all async [%l0]
    %l1, %lm1 = air.execute [%l0, %r0] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r1, %rm1 = air.execute [%l0, %r0] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l2, %lm2 = air.execute [%l1, %r1] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r2, %rm2 = air.execute [%l1, %r1] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l3, %lm3 = air.execute [%l2, %r2] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r3, %rm3 = air.execute [%l2, %r2] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l4, %lm4 = air.execute [%l3, %r3] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r4, %rm4 = air.execute [%l3, %r3] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l5, %lm5 = air.execute [%l4, %r4] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r5, %rm5 = air.execute [%l4, %r4] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l6, %lm6 = air.execute [%l5, %r5] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r6, %rm6 = air.execute [%l5, %r5] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l7, %lm7 = air.execute [%l6, %r6] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r7, %rm7 = air.execute [%l6, %r6] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l8, %lm8 = air.execute [%l7, %r7] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r8, %rm8 = air.execute [%l7, %r7] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l9, %lm9 = air.execute [%l8, %r8] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r9, %rm9 = air.execute [%l8, %r8] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l10, %lm10 = air.execute [%l9, %r9] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r10, %rm10 = air.execute [%l9, %r9] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l11, %lm11 = air.execute [%l10, %r10] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r11, %rm11 = air.execute [%l10, %r10] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l12, %lm12 = air.execute [%l11, %r11] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r12, %rm12 = air.execute [%l11, %r11] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l13, %lm13 = air.execute [%l12, %r12] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r13, %rm13 = air.execute [%l12, %r12] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l14, %lm14 = air.execute [%l13, %r13] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r14, %rm14 = air.execute [%l13, %r13] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l15, %lm15 = air.execute [%l14, %r14] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r15, %rm15 = air.execute [%l14, %r14] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l16, %lm16 = air.execute [%l15, %r15] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r16, %rm16 = air.execute [%l15, %r15] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l17, %lm17 = air.execute [%l16, %r16] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r17, %rm17 = air.execute [%l16, %r16] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l18, %lm18 = air.execute [%l17, %r17] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r18, %rm18 = air.execute [%l17, %r17] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l19, %lm19 = air.execute [%l18, %r18] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r19, %rm19 = air.execute [%l18, %r18] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l20, %lm20 = air.execute [%l19, %r19] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r20, %rm20 = air.execute [%l19, %r19] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l21, %lm21 = air.execute [%l20, %r20] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r21, %rm21 = air.execute [%l20, %r20] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l22, %lm22 = air.execute [%l21, %r21] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r22, %rm22 = air.execute [%l21, %r21] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l23, %lm23 = air.execute [%l22, %r22] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r23, %rm23 = air.execute [%l22, %r22] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l24, %lm24 = air.execute [%l23, %r23] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r24, %rm24 = air.execute [%l23, %r23] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l25, %lm25 = air.execute [%l24, %r24] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r25, %rm25 = air.execute [%l24, %r24] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l26, %lm26 = air.execute [%l25, %r25] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r26, %rm26 = air.execute [%l25, %r25] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l27, %lm27 = air.execute [%l26, %r26] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r27, %rm27 = air.execute [%l26, %r26] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l28, %lm28 = air.execute [%l27, %r27] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r28, %rm28 = air.execute [%l27, %r27] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l29, %lm29 = air.execute [%l28, %r28] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r29, %rm29 = air.execute [%l28, %r28] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %l30, %lm30 = air.execute [%l29, %r29] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %r30, %rm30 = air.execute [%l29, %r29] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %a = air.channel.put async [%l30, %r30] @channel_a[] (%arg0[] [] []) : (memref<64xi32>)
    %b = air.channel.put async [%l30, %r30] @channel_b[] (%arg1[] [] []) : (memref<64xi32>)
    %y = air.wait_all async [%a, %b]
    scf.yield %y : !air.async.token
  }
  return
}

// -----

// The put on channel_b waits on the put on channel_a only through two ops that
// are not themselves isolated, so the two puts stay in one loop.

// CHECK-LABEL: func.func @through_other_ops
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_a
// CHECK-NOT: scf.yield
// CHECK: air.channel.put{{.*}}@channel_b
// CHECK: scf.yield
// CHECK-NOT: scf.for

air.channel @channel_a [1, 1]
air.channel @channel_b [1, 1]
func.func @through_other_ops(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t0 = air.wait_all async
  %t = scf.for %i = %c0 to %c8 step %c1 iter_args(%d = %t0) -> (!air.async.token) {
    %a = air.channel.put async [%d] @channel_a[] (%arg0[] [] []) : (memref<64xi32>)
    %e1, %m1 = air.execute [%a] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %e2, %m2 = air.execute [%e1] -> (memref<8xi32, 2>) {
      %m = memref.alloc() : memref<8xi32, 2>
      air.execute_terminator %m : memref<8xi32, 2>
    }
    %b = air.channel.put async [%e2] @channel_b[] (%arg1[] [] []) : (memref<64xi32>)
    scf.yield %b : !air.async.token
  }
  return
}
