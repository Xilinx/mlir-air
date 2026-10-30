//===- hoisted_loop_multi_result_raw_dependency.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file %s -air-dma-to-channel | FileCheck %s

// The actual writer loop returns a live index and two tokens. The hoisted
// read must depend on both writer tokens, not only the placeholder loop.
// CHECK: %[[LOOP:.*]]:3 = scf.for
// CHECK: air.channel.get async {{.*}}@Drain
// CHECK: air.channel.put async [{{.*}}%[[LOOP]]#2, %[[LOOP]]#1{{.*}}] @FaIn

module {
  air.channel @Drain [1]
  air.channel @FaIn [1]
  func.func @f(%qkv: memref<256x1600xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%x=%qkv) : memref<256x1600xbf16> {
      air.segment @s args(%y=%x) : memref<256x1600xbf16> {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1s = arith.constant 1 : index
        %start = air.wait_all async
        %loop:3 = scf.for %i = %c0 to %c3 step %c1s iter_args(%base = %c0, %token = %start, %other = %start) -> (index, !air.async.token, !air.async.token) {
          %r = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%base]
          %get = air.channel.get async [%token] @Drain[%c0] (%y[%r, 0] [32, 320] [1600, 1]) : (memref<256x1600xbf16>)
          %next = arith.addi %base, %c1s : index
          scf.yield %next, %get, %other : index, !air.async.token, !air.async.token
        }
        %width = arith.muli %loop#0, %c3 : index
        %read = air.channel.put async [%loop#1] @FaIn[%c0] (%y[0, 0] [64, %width] [1600, 1]) : (memref<256x1600xbf16>)
      }
    }
    return
  }
}

// -----

// A parallel writer's reduction token is also a dependency source.
// CHECK-LABEL: func.func @parallel_writer
// CHECK: %[[PAR:.*]] = scf.parallel
// CHECK: air.channel.get async {{.*}}@Drain
// CHECK: air.channel.put async [{{.*}}%[[PAR]],{{.*}}] @FaIn

module {
  air.channel @Drain [1]
  air.channel @FaIn [1]
  func.func @parallel_writer(%qkv: memref<256x1600xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%x=%qkv) : memref<256x1600xbf16> {
      air.segment @s args(%y=%x) : memref<256x1600xbf16> {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1s = arith.constant 1 : index
        %start = air.wait_all async
        %loop = scf.parallel (%i) = (%c0) to (%c3) step (%c1s) init (%start) -> !air.async.token {
          %r = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%i]
          %get = air.channel.get async [%start] @Drain[%c0] (%y[%r, 0] [32, 320] [1600, 1]) : (memref<256x1600xbf16>)
          scf.reduce(%get : !air.async.token) {
          ^bb0(%lhs: !air.async.token, %rhs: !air.async.token):
            %both = air.wait_all async [%lhs, %rhs]
            scf.reduce.return %both : !air.async.token
          }
        }
        %read = air.channel.put async [%loop] @FaIn[%c0] (%y[0, 0] [64, 64] [1600, 1]) : (memref<256x1600xbf16>)
      }
    }
    return
  }
}
