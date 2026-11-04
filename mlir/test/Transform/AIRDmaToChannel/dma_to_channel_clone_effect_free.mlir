//===- dma_to_channel_clone_effect_free.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// The L3 side of a segment's DMA is hoisted to the launch together with the
// address arithmetic it uses. An arith.divsi by a run-time divisor has no
// memory effects but is not speculatable; it must be cloned along with its
// users rather than left in the segment.

// CHECK-LABEL: func.func @divsi_in_address
// CHECK: air.launch async {{.*}} args({{.*}}, %[[N:.*]]=%{{.*}})
// CHECK: scf.for %[[IV:[a-z0-9_]+]] =
// CHECK-NEXT: %[[DIV:.*]] = arith.divsi %[[IV]], %[[N]] : index
// CHECK-NEXT: %[[OFF:.*]] = arith.muli %[[DIV]], %{{.*}} : index
// CHECK: air.channel.put {{.*}}(%{{.*}}[%[[OFF]]]
// CHECK: air.segment
// CHECK-NOT: arith.divsi
// CHECK: air.channel.get
module {
  func.func @divsi_in_address(%arg0: memref<4096xbf16>, %n: index) {
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
          %d = arith.divsi %i, %sn : index
          %off = arith.muli %d, %c64 : index
          %3 = air.dma_memcpy_nd async [%it] (%buf[] [] [], %sa[%off] [%c64] [%c1_0]) {id = 1 : i32} : (memref<64xbf16, 1>, memref<4096xbf16>)
          scf.yield %3 : !air.async.token
        }
      }
    }
    return
  }
}
