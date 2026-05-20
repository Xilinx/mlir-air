//===- air_split_l2_memref_launch_cap_s2mm.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1 max-launch-channels-mm2s=4 max-launch-channels-s2mm=4" 2>&1 | FileCheck %s

// Mirror of the MM2S launch-cap test with directions reversed. The per-col L2
// allocs have multiple herd-side puts (compute → L2 via chan_in) and a single
// L3-bound get (L2 → L3 via chan_out). chan_out [4] has 4 launch-level GETS,
// so the launch S2MM count is 4. Splitting 2× would lift it to 8 > cap=4 and
// must be rejected via the S2MM-direction branch of the cap check.

// CHECK-COUNT-4: remark: air-split-l2-memref: skipping split (factor=2) on memref @chan_out to avoid pushing launch S2MM endpoint count from 4 to 8 (cap=4)
// CHECK: air.channel @chan_out [4]
// CHECK-NOT: air.channel @chan_out [2, 4]
// CHECK-LABEL: func.func @test_s2mm
// CHECK-COUNT-4: memref.alloc() : memref<8x1024xbf16, 1 : i32>

air.channel @chan_in [4, 2]
air.channel @chan_out [4]
func.func @test_s2mm(%arg0: memref<128x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg1) in (%arg2=%c1) args(%arg3=%arg0) : memref<128x1024xbf16> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    // 4 launch-level gets on chan_out → S2MM raw endpoints = 4.
    %g0 = air.channel.get async @chan_out[%c0] (%arg3[%c0, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 1 : i32} : (memref<128x1024xbf16>)
    %g1 = air.channel.get async @chan_out[%c1_l] (%arg3[%c8, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 2 : i32} : (memref<128x1024xbf16>)
    %g2 = air.channel.get async @chan_out[%c2] (%arg3[%c0, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 3 : i32} : (memref<128x1024xbf16>)
    %g3 = air.channel.get async @chan_out[%c3] (%arg3[%c8, %c0] [%c8, %c1024] [%c1024, %c1_l]) {id = 4 : i32} : (memref<128x1024xbf16>)
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
      // Multiple gets into each L2 alloc from herd-side puts (S2MM side > 1).
      %p00 = air.channel.get async [%async_token_a] @chan_in[%c0_0, %c0_0] (%a0[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p01 = air.channel.get async [%async_token_a] @chan_in[%c0_0, %c1_0] (%a0[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p10 = air.channel.get async [%async_token_b] @chan_in[%c1_0, %c0_0] (%a1[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p11 = air.channel.get async [%async_token_b] @chan_in[%c1_0, %c1_0] (%a1[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p20 = air.channel.get async [%async_token_c] @chan_in[%c2_0, %c0_0] (%a2[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p21 = air.channel.get async [%async_token_c] @chan_in[%c2_0, %c1_0] (%a2[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p30 = air.channel.get async [%async_token_d] @chan_in[%c3_0, %c0_0] (%a3[%c0_0, %c0_0] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      %p31 = air.channel.get async [%async_token_d] @chan_in[%c3_0, %c1_0] (%a3[%c0_0, %c512] [%c8_0, %c512] [%c1024_0, %c1_0]) : (memref<8x1024xbf16, 1 : i32>)
      // Single put per L2 alloc → chan_out (MM2S side = 1).
      %op0 = air.channel.put async [%p00, %p01] @chan_out[%c0_0] (%a0[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %op1 = air.channel.put async [%p10, %p11] @chan_out[%c1_0] (%a1[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %op2 = air.channel.put async [%p20, %p21] @chan_out[%c2_0] (%a2[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %op3 = air.channel.put async [%p30, %p31] @chan_out[%c3_0] (%a3[] [] []) : (memref<8x1024xbf16, 1 : i32>)
      %c4_0 = arith.constant 4 : index
      %h = air.herd @h async tile (%tx, %ty) in (%sx=%c4_0, %sy=%c2_0) args(%a=%a0) : memref<8x1024xbf16, 1 : i32> {
        %async_token_h, %r = air.execute -> (memref<8x512xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<8x512xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<8x512xbf16, 2 : i32>
        }
        %ph = air.channel.put async [%async_token_h] @chan_in[%tx, %ty] (%r[] [] []) : (memref<8x512xbf16, 2 : i32>)
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
