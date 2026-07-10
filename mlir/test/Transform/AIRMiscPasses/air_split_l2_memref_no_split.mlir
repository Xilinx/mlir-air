//===- air_split_l2_memref_no_split.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// Opt-out: an L2 alloc carrying `air.no_split` is left intact by the L2-memref
// splitting pass, even when its channel access pattern would otherwise be
// split. Hand-written aggregator buffers rely on this to keep their single
// wide L2 buffer (splitting would multiply launch-level shim endpoints and
// overflow the memtile BD limit). Without honoring the attribute the pass
// splits the 256x256 buffer into four 64x256 fragments behind a new split
// channel.

// A split would introduce @channel_2 at module scope, before @test0 - assert
// its absence before the label so the CHECK-NOT actually covers module scope.
// CHECK-NOT: air.channel @channel_2
// CHECK-LABEL: func.func @test0
// CHECK: memref.alloc() {air.no_split} : memref<256x256xbf16, 1 : i32>
// CHECK-NOT: memref<64x256xbf16, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
air.channel @channel_1 [1, 1]
air.channel @channel_0 [4, 4]
func.func @test0(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg2) : memref<512x512xbf16> attributes {id = 1 : i32} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %async_token, %results = air.execute -> (index) {
      %3 = affine.apply #map()[%arg3]
      air.execute_terminator %3 : index
    }
    %async_token_0, %results_1 = air.execute -> (index) {
      %3 = affine.apply #map()[%arg4]
      air.execute_terminator %3 : index
    }
    %1 = air.channel.get async [%async_token, %async_token_0]  @channel_1[] (%arg7[%results, %results_1] [%c256, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
    %2 = air.segment @segment_0 async  {
      %c64 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256_3 = arith.constant 256 : index
      %3 = air.wait_all async 
      %4 = air.wait_all async 
      %async_token_4, %results_5 = air.execute -> (memref<256x256xbf16, 1 : i32>) {
        %alloc = memref.alloc() {air.no_split} : memref<256x256xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<256x256xbf16, 1 : i32>
      }
      %5 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c4, %c4) step (%c1_2, %c1_2) init (%async_token_4) -> !air.async.token {
        %async_token_7, %results_8 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg8]
          air.execute_terminator %9 : index
        }
        %async_token_9, %results_10 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg9]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.get async [%async_token_4, %async_token_9, %async_token_7]  @channel_0[%arg8, %arg9] (%results_5[%results_8, %results_10] [%c64, %c64] [%c256_3, %c1_2]) {id = 24 : i32} : (memref<256x256xbf16, 1 : i32>)
        scf.reduce(%8 : !air.async.token) {
        ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
          %9 = air.wait_all async [%arg10, %arg11] 
          scf.reduce.return %9 : !air.async.token
        }
      }
      %6 = air.herd @herd_0 async [%async_token_4]  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c64_7 = arith.constant 64 : index
        %c256_8 = arith.constant 256 : index
        %c4_9 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %c0_11 = arith.constant 0 : index
        %async_token_12, %results_13 = air.execute -> (memref<16x16x4x4xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<16x16x4x4xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<16x16x4x4xbf16, 2 : i32>
        }
        %8 = air.channel.put async [%async_token_12]  @channel_0[%arg8, %arg9] (%results_13[%c0_11, %c0_11, %c0_11] [%c64_7, %c16, %c4_9] [%c4_9, %c256_8, %c1_10]) {id = 41 : i32} : (memref<16x16x4x4xbf16, 2 : i32>)
        %async_token_14 = air.execute [%8] {
          memref.dealloc %results_13 : memref<16x16x4x4xbf16, 2 : i32>
        }
      }
      %7 = air.channel.put async [%3, %4, %6]  @channel_1[] (%results_5[] [] []) {id = 42 : i32} : (memref<256x256xbf16, 1 : i32>)
      %async_token_6 = air.execute [%7] {
        memref.dealloc %results_5 : memref<256x256xbf16, 1 : i32>
      }
    }
  }
  return
}


// -----

// Multiple-in-single-out aggregator (the shape a per-column output
// assembler uses): each L2 buffer is filled by several channel gets and
// drained by one put. With air.no_split the wide 8x1024 buffers are kept
// intact; without it each is split into 8x512 fragments.

// CHECK-LABEL: func.func @test_s2mm
// CHECK: memref.alloc() {air.no_split} : memref<8x1024xbf16, 1 : i32>
// CHECK-NOT: memref<8x512xbf16, 1 : i32>

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
        %alloc = memref.alloc() {air.no_split} : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %async_token_b, %a1 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() {air.no_split} : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %async_token_c, %a2 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() {air.no_split} : memref<8x1024xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<8x1024xbf16, 1 : i32>
      }
      %async_token_d, %a3 = air.execute -> (memref<8x1024xbf16, 1 : i32>) {
        %alloc = memref.alloc() {air.no_split} : memref<8x1024xbf16, 1 : i32>
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

// -----

// Baseline (no opt-out): the same access pattern as @test0 without the
// `air.no_split` attribute must still split, so @test0's CHECK-NOTs cannot
// pass vacuously. The 256x256 buffer becomes four 64x256 fragments.

// CHECK-LABEL: func.func @test0_split
// CHECK: memref<64x256xbf16, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
air.channel @channel_1 [1, 1]
air.channel @channel_0 [4, 4]
func.func @test0_split(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg2) : memref<512x512xbf16> attributes {id = 1 : i32} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %async_token, %results = air.execute -> (index) {
      %3 = affine.apply #map()[%arg3]
      air.execute_terminator %3 : index
    }
    %async_token_0, %results_1 = air.execute -> (index) {
      %3 = affine.apply #map()[%arg4]
      air.execute_terminator %3 : index
    }
    %1 = air.channel.get async [%async_token, %async_token_0]  @channel_1[] (%arg7[%results, %results_1] [%c256, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
    %2 = air.segment @segment_0 async  {
      %c64 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256_3 = arith.constant 256 : index
      %3 = air.wait_all async
      %4 = air.wait_all async
      %async_token_4, %results_5 = air.execute -> (memref<256x256xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<256x256xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<256x256xbf16, 1 : i32>
      }
      %5 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c4, %c4) step (%c1_2, %c1_2) init (%async_token_4) -> !air.async.token {
        %async_token_7, %results_8 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg8]
          air.execute_terminator %9 : index
        }
        %async_token_9, %results_10 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg9]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.get async [%async_token_4, %async_token_9, %async_token_7]  @channel_0[%arg8, %arg9] (%results_5[%results_8, %results_10] [%c64, %c64] [%c256_3, %c1_2]) {id = 24 : i32} : (memref<256x256xbf16, 1 : i32>)
        scf.reduce(%8 : !air.async.token) {
        ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
          %9 = air.wait_all async [%arg10, %arg11]
          scf.reduce.return %9 : !air.async.token
        }
      }
      %6 = air.herd @herd_0 async [%async_token_4]  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c64_7 = arith.constant 64 : index
        %c256_8 = arith.constant 256 : index
        %c4_9 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %c0_11 = arith.constant 0 : index
        %async_token_12, %results_13 = air.execute -> (memref<16x16x4x4xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<16x16x4x4xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<16x16x4x4xbf16, 2 : i32>
        }
        %8 = air.channel.put async [%async_token_12]  @channel_0[%arg8, %arg9] (%results_13[%c0_11, %c0_11, %c0_11] [%c64_7, %c16, %c4_9] [%c4_9, %c256_8, %c1_10]) {id = 41 : i32} : (memref<16x16x4x4xbf16, 2 : i32>)
        %async_token_14 = air.execute [%8] {
          memref.dealloc %results_13 : memref<16x16x4x4xbf16, 2 : i32>
        }
      }
      %7 = air.channel.put async [%3, %4, %6]  @channel_1[] (%results_5[] [] []) {id = 42 : i32} : (memref<256x256xbf16, 1 : i32>)
      %async_token_6 = air.execute [%7] {
        memref.dealloc %results_5 : memref<256x256xbf16, 1 : i32>
      }
    }
  }
  return
}

// -----

// Opt-out placed on the enclosing air.execute wrapper (rather than the alloc)
// is also honored: the pass checks the alloc's parent air.execute for
// `air.no_split`. The 256x256 buffer is kept intact.

// CHECK-NOT: air.channel @channel_2
// CHECK-LABEL: func.func @test_exec_attr
// CHECK: air.execute
// CHECK: memref.alloc() : memref<256x256xbf16, 1 : i32>
// CHECK: {air.no_split}
// CHECK-NOT: memref<64x256xbf16, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
air.channel @channel_1 [1, 1]
air.channel @channel_0 [4, 4]
func.func @test_exec_attr(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg2) : memref<512x512xbf16> attributes {id = 1 : i32} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %async_token, %results = air.execute -> (index) {
      %3 = affine.apply #map()[%arg3]
      air.execute_terminator %3 : index
    }
    %async_token_0, %results_1 = air.execute -> (index) {
      %3 = affine.apply #map()[%arg4]
      air.execute_terminator %3 : index
    }
    %1 = air.channel.get async [%async_token, %async_token_0]  @channel_1[] (%arg7[%results, %results_1] [%c256, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
    %2 = air.segment @segment_0 async  {
      %c64 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256_3 = arith.constant 256 : index
      %3 = air.wait_all async
      %4 = air.wait_all async
      %async_token_4, %results_5 = air.execute -> (memref<256x256xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<256x256xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<256x256xbf16, 1 : i32>
      } {air.no_split}
      %5 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c4, %c4) step (%c1_2, %c1_2) init (%async_token_4) -> !air.async.token {
        %async_token_7, %results_8 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg8]
          air.execute_terminator %9 : index
        }
        %async_token_9, %results_10 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg9]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.get async [%async_token_4, %async_token_9, %async_token_7]  @channel_0[%arg8, %arg9] (%results_5[%results_8, %results_10] [%c64, %c64] [%c256_3, %c1_2]) {id = 24 : i32} : (memref<256x256xbf16, 1 : i32>)
        scf.reduce(%8 : !air.async.token) {
        ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
          %9 = air.wait_all async [%arg10, %arg11]
          scf.reduce.return %9 : !air.async.token
        }
      }
      %6 = air.herd @herd_0 async [%async_token_4]  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c64_7 = arith.constant 64 : index
        %c256_8 = arith.constant 256 : index
        %c4_9 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %c0_11 = arith.constant 0 : index
        %async_token_12, %results_13 = air.execute -> (memref<16x16x4x4xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<16x16x4x4xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<16x16x4x4xbf16, 2 : i32>
        }
        %8 = air.channel.put async [%async_token_12]  @channel_0[%arg8, %arg9] (%results_13[%c0_11, %c0_11, %c0_11] [%c64_7, %c16, %c4_9] [%c4_9, %c256_8, %c1_10]) {id = 41 : i32} : (memref<16x16x4x4xbf16, 2 : i32>)
        %async_token_14 = air.execute [%8] {
          memref.dealloc %results_13 : memref<16x16x4x4xbf16, 2 : i32>
        }
      }
      %7 = air.channel.put async [%3, %4, %6]  @channel_1[] (%results_5[] [] []) {id = 42 : i32} : (memref<256x256xbf16, 1 : i32>)
      %async_token_6 = air.execute [%7] {
        memref.dealloc %results_5 : memref<256x256xbf16, 1 : i32>
      }
    }
  }
  return
}
