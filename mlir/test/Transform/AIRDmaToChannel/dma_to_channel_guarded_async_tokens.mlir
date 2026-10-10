//===- dma_to_channel_guarded_async_tokens.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// A guarded L2 to L1 DMA whose scf.if yields two tokens, and whose else branch
// passes the alloc's token straight through. Each hoisted branch yields what
// it carried over, and the hoisted loop waits on both results.

// CHECK-LABEL: func.func @two_tokens
// CHECK: air.segment
// CHECK: scf.parallel
// CHECK: %[[ALLOC:.*]] = air.wait_all async
// CHECK: %[[IF:.*]]:2 = scf.if %{{.*}} -> (!air.async.token, !air.async.token) {
// CHECK: air.channel.put async {{.*}}@channel_1
// CHECK: scf.yield
// CHECK: } else {
// CHECK: %[[E0:.*]] = air.wait_all async [%[[ALLOC]]]
// CHECK: %[[E1:.*]] = air.wait_all async [%[[ALLOC]]]
// CHECK: scf.yield %[[E0]], %[[E1]]
// CHECK: }
// CHECK: %[[R:.*]] = air.wait_all async [%[[IF]]#1, %[[IF]]#0]
// CHECK: scf.reduce(%[[R]]

module {
  func.func @two_tokens(%arg0: memref<16x960xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1) in (%arg2=%c1) args(%arg3=%arg0) : memref<16x960xbf16> attributes {id = 3 : i32} {
      %1 = air.segment @seg async  args(%arg4=%arg3) : memref<16x960xbf16> attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<16x960xbf16, 1>) {
          %alloc = memref.alloc() : memref<16x960xbf16, 1>
          air.execute_terminator %alloc : memref<16x960xbf16, 1>
        } {id = 1 : i32}
        %2 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg4[] [] []) {id = 1 : i32} : (memref<16x960xbf16, 1>, memref<16x960xbf16>)
        %3 = air.herd @herd_0 async [%2]  tile (%arg5, %arg6) in (%arg7=%c1_0, %arg8=%c4) args(%arg9=%results) : memref<16x960xbf16, 1> attributes {id = 1 : i32} {
          %c0 = arith.constant 0 : index
          %c1_2 = arith.constant 1 : index
          %c3 = arith.constant 3 : index
          %c16 = arith.constant 16 : index
          %c192 = arith.constant 192 : index
          %c256 = arith.constant 256 : index
          %c960 = arith.constant 960 : index
          %async_token_3, %results_4 = air.execute -> (memref<16x256xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x256xbf16, 2>
            air.execute_terminator %alloc : memref<16x256xbf16, 2>
          } {id = 2 : i32}
          %4 = arith.muli %arg6, %c256 : index
          %5 = arith.cmpi eq, %arg6, %c3 : index
          %6 = air.wait_all async [%async_token_3]  {id = 1 : i32}
          %7:2 = scf.if %5 -> (!air.async.token, !air.async.token) {
            %8 = air.dma_memcpy_nd async [%async_token_3] (%results_4[] [] [], %arg9[%c0, %4] [%c16, %c192] [%c960, %c1_2]) {id = 2 : i32, pad_after = array<i32: 0, 64>, pad_before = array<i32: 0, 0>} : (memref<16x256xbf16, 2>, memref<16x960xbf16, 1>)
            %9 = air.wait_all async [%8]  {id = 2 : i32}
            scf.yield %9, %8 : !air.async.token, !air.async.token
          } else {
            scf.yield %async_token_3, %async_token_3 : !air.async.token, !air.async.token
          }
          %async_token_5 = air.execute [%7#0, %7#1] {
            memref.dealloc %results_4 : memref<16x256xbf16, 2>
          } {id = 3 : i32}
        }
        %async_token_1 = air.execute [%3] {
          memref.dealloc %results : memref<16x960xbf16, 1>
        } {id = 4 : i32}
      }
    }
    return
  }
}

