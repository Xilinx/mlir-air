//===- air_split_l2_memref_launch_cap.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1 max-launch-channels-mm2s=4 max-launch-channels-s2mm=4" | FileCheck %s

// chan_in [4] has 4 launch-level puts (1/col). Splitting the L2 buf 2x would
// prepend a 2 to the channel shape, doubling launch-level MM2S endpoint count
// to 8. With the cap set to 4, the pass must skip the split, leaving chan_in
// unchanged and the per-col 8x1024 L2 bufs intact.

// CHECK: air.channel @chan_in [4]
// CHECK-NOT: [2, 4]
// CHECK-LABEL: func.func @test
// CHECK: air.channel.put {{.*}} @chan_in[%{{.*}}]
// CHECK: air.segment
// CHECK-COUNT-4: memref.alloc() : memref<8x1024xbf16, 1 : i32>
// CHECK-COUNT-4: air.channel.get {{.*}} @chan_in

#map = affine_map<()[s0] -> (s0 * 32)>
air.channel @chan_in [4]
air.channel @chan_out [4, 2]
func.func @test(%arg0: memref<128x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg1) in (%arg2=%c1) args(%arg3=%arg0) : memref<128x1024xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %1 = air.channel.put async @chan_in[%c0] (%arg3[%c0, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 1 : i32} : (memref<128x1024xbf16>)
    %2 = air.channel.put async @chan_in[%c1_l] (%arg3[%c8, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 2 : i32} : (memref<128x1024xbf16>)
    %3 = air.channel.put async @chan_in[%c2] (%arg3[%c0, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 3 : i32} : (memref<128x1024xbf16>)
    %4 = air.channel.put async @chan_in[%c3] (%arg3[%c8, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 4 : i32} : (memref<128x1024xbf16>)
    %5 = air.segment @seg async {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %c3_0 = arith.constant 3 : index
      %c8_0 = arith.constant 8 : index
      %c512 = arith.constant 512 : index
      %c1024_0 = arith.constant 1024 : index
      %async_token_a, %a0 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %async_token_b, %a1 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %async_token_c, %a2 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %async_token_d, %a3 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %g0 = air.channel.get async [%async_token_a] @chan_in[%c0_0] (%a0[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %g1 = air.channel.get async [%async_token_b] @chan_in[%c1_0] (%a1[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %g2 = air.channel.get async [%async_token_c] @chan_in[%c2_0] (%a2[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %g3 = air.channel.get async [%async_token_d] @chan_in[%c3_0] (%a3[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %p00 = air.channel.put async [%g0] @chan_out[%c0_0, %c0_0] (%a0[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p01 = air.channel.put async [%g0] @chan_out[%c0_0, %c1_0] (%a0[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p10 = air.channel.put async [%g1] @chan_out[%c1_0, %c0_0] (%a1[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p11 = air.channel.put async [%g1] @chan_out[%c1_0, %c1_0] (%a1[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p20 = air.channel.put async [%g2] @chan_out[%c2_0, %c0_0] (%a2[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p21 = air.channel.put async [%g2] @chan_out[%c2_0, %c1_0] (%a2[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p30 = air.channel.put async [%g3] @chan_out[%c3_0, %c0_0] (%a3[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p31 = air.channel.put async [%g3] @chan_out[%c3_0, %c1_0] (%a3[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %c4_0 = arith.constant 4 : index
      %h = air.herd @h async tile (%tx, %ty) in (%sx=%c4_0, %sy=%c2_0) args(%a=%a0, %b=%a1, %c=%a2, %d=%a3) : memref<8x1024xbf16, 1 : i32>, memref<8x1024xbf16, 1 : i32>, memref<8x1024xbf16, 1 : i32>, memref<8x1024xbf16, 1 : i32> {
        %async_token_h, %r = air.execute -> (memref<8x512xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<8x512xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<8x512xbf16, 2 : i32>
        }
        %gh = air.channel.get async [%async_token_h] @chan_out[%tx, %ty] (%r[] [] []) : (memref<8x512xbf16, 2 : i32>)
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
