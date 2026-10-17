//===- dma_to_channel_herd_token_order.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// A herd inside a K loop whose DMAs are hoisted into an scf.parallel: the loop
// carried token must still cover the herd, not only the hoisted scf.parallel
// (which tracks just the L2-side channel puts). Otherwise the drain herd after
// the loop can read the L1 accumulator before the last compute finished.

// CHECK-LABEL: func.func @herd_token_order
// CHECK: scf.for
// CHECK: %[[PAR:.*]] = scf.parallel
// CHECK: %[[COMPUTE:.*]] = air.herd @compute async
// CHECK: %[[JOIN:.*]] = air.wait_all async [%[[PAR]], %[[COMPUTE]]]
// CHECK: scf.yield
// CHECK: air.herd @drain async
func.func @herd_token_order(%arg0: memref<64x64xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%lx, %ly) in (%s0=%c1, %s1=%c1) args(%a=%arg0) : memref<64x64xbf16> {
    %1 = air.segment @seg async args(%sa=%a) : memref<64x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %t_l2, %l2 = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %t_acc, %acc = air.execute -> (memref<32x32xbf16, 2>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 2>
        air.execute_terminator %alloc : memref<32x32xbf16, 2>
      }
      %init = air.wait_all async [%t_l2, %t_acc]
      %k = scf.for %iv = %c0 to %c4 step %c1_0 iter_args(%tok = %init) -> (!air.async.token) {
        %h0 = air.herd @compute async [%tok] tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%c=%acc, %src=%l2) : memref<32x32xbf16, 2>, memref<32x32xbf16, 1> {
          %t_in, %in = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %d = air.dma_memcpy_nd async [%t_in] (%in[] [] [], %src[] [] []) : (memref<32x32xbf16, 2>, memref<32x32xbf16, 1>)
          %t_mac = air.execute [%d] {
            memref.copy %in, %c : memref<32x32xbf16, 2> to memref<32x32xbf16, 2>
          }
        }
        scf.yield %h0 : !air.async.token
      }
      %h1 = air.herd @drain async [%k] tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%c=%acc, %dst=%l2) : memref<32x32xbf16, 2>, memref<32x32xbf16, 1> {
        %d = air.dma_memcpy_nd async (%dst[] [] [], %c[] [] []) : (memref<32x32xbf16, 1>, memref<32x32xbf16, 2>)
      }
    }
  }
  return
}
