//===- dma_to_channel_guarded_not_speculatable.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not air-opt %s -air-dma-to-channel 2>&1 | FileCheck %s

// The DMA's offset divides by %n under an affine.if that checks %n >= 1.
// Hoisting flattens the affine.if, so a clone of the division would run
// without that check. The division is not cloned; the pass reports it.
// CHECK: error: 'arith.divsi' op cannot be hoisted out of the condition that guards it: it is not speculatable

#set = affine_set<()[s0] : (s0 - 1 >= 0)>
module {
  func.func @guarded_divsi(%arg0: memref<4096xbf16>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l0) in (%ls0=%c1) args(%a=%arg0, %ln=%n) : memref<4096xbf16>, index {
      %1 = air.segment @seg async args(%sa=%a, %sn=%ln) : memref<4096xbf16>, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %i = %c0 to %c8 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %r = affine.if #set()[%sn] -> !air.async.token {
            %d = arith.divsi %i, %sn : index
            %off = arith.muli %d, %c64 : index
            %3 = air.dma_memcpy_nd async [%it] (%buf[] [] [], %sa[%off] [%c64] [%c1_0]) {id = 1 : i32} : (memref<64xbf16, 1>, memref<4096xbf16>)
            affine.yield %3 : !air.async.token
          } else {
            %w = air.wait_all async [%it]
            affine.yield %w : !air.async.token
          }
          scf.yield %r : !air.async.token
        }
      }
    }
    return
  }
}
