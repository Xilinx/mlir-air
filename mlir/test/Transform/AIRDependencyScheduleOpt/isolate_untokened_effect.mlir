//===- isolate_untokened_effect.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests | FileCheck %s

// A loop with two independent transfers and an inner loop with effects but no
// token. Split, the inner loop would belong to neither part and be dropped,
// so the loop stays whole.

// CHECK-LABEL: func.func @untokened_effect
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@a
// CHECK: air.channel.put{{.*}}@b
// CHECK: scf.for
// CHECK: memref.store
// CHECK-NOT: scf.for {{.*}} iter_args

module {
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @untokened_effect(%n: i32) {
    %c1 = arith.constant 1 : index
    %t = air.launch async (%la) in (%ls=%c1) args(%ln=%n) : i32 {
      %s = air.segment @seg async args(%sn=%ln) : i32 {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %ta, %ba = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %tb, %bb = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %tc, %bc = air.execute -> (memref<4xi32, 1>) {
          %m = memref.alloc() : memref<4xi32, 1>
          air.execute_terminator %m : memref<4xi32, 1>
        }
        %w = air.wait_all async [%ta, %tb, %tc]
        %r = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
          %pa = air.channel.put async [%d] @a[] (%ba[] [] []) : (memref<64xbf16, 1>)
          %pb = air.channel.put async [%d] @b[] (%bb[] [] []) : (memref<64xbf16, 1>)
          scf.for %j = %c0_i32 to %sn step %c1_i32 : i32 {
            memref.store %j, %bc[%i] : memref<4xi32, 1>
          }
          %y = air.wait_all async [%pa, %pb]
          scf.yield %y : !air.async.token
        }
      }
    }
    return
  }
}
