//===- dma_to_channel_copy_out_broadcast_set.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel 2>&1 | FileCheck %s

// A broadcast has one source outside and many receivers inside. A copy out of
// the tiles under a broadcast_set is not one: every tile in the set sends its
// own transfer. The channel is the herd's shape and both ends are indexed by
// tile, as for a copy with no set.

// CHECK: warning: Attribute broadcast_set is set on a copy out of the inner memory space
// CHECK: air.channel @[[C:.*]] [1, 2]
// CHECK-NOT: broadcast_shape
// CHECK-LABEL: func.func @copy_out
// CHECK: scf.parallel (%[[X:.*]], %[[Y:.*]]) =
// CHECK: air.channel.get @[[C]][%[[X]], %[[Y]]]
// CHECK: air.herd
// CHECK: air.channel.put @[[C]][%{{.*}}, %{{.*}}]
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
func.func @copy_out(%arg0: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  air.launch (%lx) in (%sx = %c1) args(%out = %arg0) : memref<64xi32> {
    air.segment @seg args(%o = %out) : memref<64xi32> {
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c2_0) args(%d = %o) : memref<64xi32> {
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %buf = memref.alloc() : memref<32xi32, 2>
        affine.if #set()[%tx, %ty] {
          air.dma_memcpy_nd (%d[%c0] [%c32] [%c1_1], %buf[] [] []) {broadcast_set = #set} : (memref<64xi32>, memref<32xi32, 2>)
        }
        memref.dealloc %buf : memref<32xi32, 2>
      }
    }
  }
  return
}
