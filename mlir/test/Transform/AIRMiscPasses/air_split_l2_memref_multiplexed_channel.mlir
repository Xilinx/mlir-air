//===- air_split_l2_memref_multiplexed_channel.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" | FileCheck %s

// One channel, @ab_out, carries transfers out of two L2 buffers. Buffer A is
// sent out one row per channel instance, so A is split by row. Buffer B
// shares the channel as a flat 1-D transfer. To decide how A splits, the pass
// looks at every put on @ab_out, including the put of B. The access pattern
// of B has no dimension that walks the rows of A (no stride of 64), so that
// put does not constrain A. A is still split into two rows. B has only one
// put and is left intact.

// CHECK-LABEL: func.func @multiplexed_channel
// CHECK-NOT: memref.alloc() : memref<2x64xbf16, 1>
// CHECK: %{{.*}}, %[[A0:.*]] = air.execute -> (memref<1x64xbf16, 1>)
// CHECK: %{{.*}}, %[[A1:.*]] = air.execute -> (memref<1x64xbf16, 1>)
// CHECK: %{{.*}}, %[[B:.*]] = air.execute -> (memref<128xbf16, 1>)
// CHECK: air.channel.put {{.*}} @ab_out[%{{.*}}, %{{.*}}] (%[[A0]]{{.*}}) {{.*}} : (memref<1x64xbf16, 1>)
// CHECK: air.channel.put {{.*}} @ab_out[%{{.*}}, %{{.*}}] (%[[A1]]{{.*}}) {{.*}} : (memref<1x64xbf16, 1>)
// CHECK: air.channel.put {{.*}} @ab_out[%{{.*}}, %{{.*}}] (%[[B]]{{.*}}) {{.*}} : (memref<128xbf16, 1>)

air.channel @a_in [1, 1]
air.channel @b_in [1, 1]
air.channel @ab_out [2, 1]
func.func @multiplexed_channel(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%l) in (%ls=%c1) args(%a3=%arg0, %b3=%arg1) : memref<128xbf16>, memref<128xbf16> {
    %c0_l = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c128_l = arith.constant 128 : index
    %put_a = air.channel.put async @a_in[] (%a3[%c0_l] [%c128_l] [%c1_l]) : (memref<128xbf16>)
    %put_b = air.channel.put async @b_in[] (%b3[%c0_l] [%c128_l] [%c1_l]) : (memref<128xbf16>)
    %1 = air.segment @seg async {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %ta, %buf_a = air.execute -> (memref<2x64xbf16, 1>) {
        %a = memref.alloc() : memref<2x64xbf16, 1>
        air.execute_terminator %a : memref<2x64xbf16, 1>
      }
      %tb, %buf_b = air.execute -> (memref<128xbf16, 1>) {
        %a = memref.alloc() : memref<128xbf16, 1>
        air.execute_terminator %a : memref<128xbf16, 1>
      }
      %get_a = air.channel.get async [%ta] @a_in[] (%buf_a[] [] []) : (memref<2x64xbf16, 1>)
      %get_b = air.channel.get async [%tb] @b_in[] (%buf_b[] [] []) : (memref<128xbf16, 1>)
      // A: one row per instance of @ab_out.
      %put_a0 = air.channel.put async [%get_a] @ab_out[%c0, %c0] (%buf_a[%c0, %c0] [%c1_s, %c64] [%c64, %c1_s]) : (memref<2x64xbf16, 1>)
      %put_a1 = air.channel.put async [%get_a] @ab_out[%c1_s, %c0] (%buf_a[%c1_s, %c0] [%c1_s, %c64] [%c64, %c1_s]) : (memref<2x64xbf16, 1>)
      // B: shares the same channel, flat, with no stride of 64.
      %put_b0 = air.channel.put async [%get_b, %put_a0] @ab_out[%c0, %c0] (%buf_b[%c0] [%c64] [%c1_s]) : (memref<128xbf16, 1>)
      %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c1_s) {
        %tl, %l1 = air.execute -> (memref<64xbf16, 2>) {
          %a = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %a : memref<64xbf16, 2>
        }
        %g0 = air.channel.get async [%tl] @ab_out[%tx, %ty] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g1 = air.channel.get async [%g0] @ab_out[%tx, %ty] (%l1[] [] []) : (memref<64xbf16, 2>)
        %dl = air.execute [%g1] {
          memref.dealloc %l1 : memref<64xbf16, 2>
        }
      }
    }
  }
  return
}
