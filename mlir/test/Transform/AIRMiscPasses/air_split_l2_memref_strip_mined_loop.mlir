//===- air_split_l2_memref_strip_mined_loop.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="max-launch-channels-mm2s=16 max-launch-channels-s2mm=16" --split-input-file | FileCheck %s

// A loop over the split dimension means one of two things, and the pass has to
// tell them apart from the access pattern alone.
//
//   * Distribution: each trip takes one slice, trips are `step` apart. The
//     partition then holds slices that are NOT adjacent in L3, so the L3-side
//     transfer grows a dimension to gather them.
//   * Strip-mining: each trip takes `step` adjacent slices. The trips abut, the
//     partition is a contiguous run, and the L3 side must stay as it was.
//
// Both loops carry a non-unit step, so the step alone does not distinguish them
// -- what does is how far a trip advances the offset relative to how much it
// takes. Gathering a strip-mined run puts the wrong rows in the partition: the
// bf16 GEMV example at herd_m=8, m_input=4 came back with 3 of every 4 output
// rows wrong.
//
// The advance is not the step either. The same strip-mined access is written
// `for j = 0 to 8 step 4` with the offset folded into an affine.apply beside the
// herd coordinate, or `for jj = 0 to 2 step 1` with the tile height in the map;
// the first carries the height in the step, the second in the map.

// Strip-mining: `for %arg11 = 0 to 8 step 4` where the trip takes 4 of the 8
// rows the partition owns. Contiguous, so the L3 put keeps its 2-D
// [8, 2048] [2048, 1] pattern and the loop keeps its step.

// CHECK-LABEL: func.func @strip_mined
// CHECK: air.channel.put{{.*}}@channel_1{{.*}}[%c8{{[_0-9]*}}, %c2048{{[_0-9]*}}] [%c2048{{[_0-9]*}}, %c1{{[_0-9]*}}]
// CHECK-COUNT-8: memref.alloc() : memref<8x2048xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<64x2048xbf16, 1 : i32>
// CHECK: scf.for %{{.*}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 * 8 + s1)>
module {
  air.channel @channel_0 []
  air.channel @channel_2 [8, 1]
  func.func @strip_mined(%arg0: memref<16384x2048xbf16>) {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c256, %arg6=%c1) args(%arg7=%arg0) : memref<16384x2048xbf16> attributes {id = 1 : i32} {
      %1 = affine.apply #map()[%arg3]
      %2 = air.channel.put async  @channel_0[] (%arg7[%1, 0] [64, 2048] [2048, 1]) {id = 1 : i32} : (memref<16384x2048xbf16>)
      %6 = air.segment @seg async  attributes {id = 2 : i32} {
        %c0_0 = arith.constant 0 : index
        %c4_1 = arith.constant 4 : index
        %c1_2 = arith.constant 1 : index
        %c8_3 = arith.constant 8 : index
        %async_token, %results = air.execute -> (memref<64x2048xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x2048xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x2048xbf16, 1 : i32>
        }
        %7 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 4 : i32} : (memref<64x2048xbf16, 1 : i32>)
        %9 = scf.parallel (%arg10) = (%c0_0) to (%c8_3) step (%c1_2) init (%7) -> !air.async.token {
          %14 = scf.for %arg11 = %c0_0 to %c8_3 step %c4_1 iter_args(%arg12 = %7) -> (!air.async.token) {
            %15 = affine.apply #map1()[%arg10, %arg11]
            %16 = air.channel.put async [%arg12]  @channel_2[%arg10, %c0_0] (%results[%15, 0] [4, 2048] [2048, 1]) {id = 5 : i32} : (memref<64x2048xbf16, 1 : i32>)
            scf.yield %16 : !air.async.token
          }
          scf.reduce(%14 : !air.async.token) {
          ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
            %15 = air.wait_all async [%arg11, %arg12]
            scf.reduce.return %15 : !air.async.token
          }
        }
        %11 = air.herd @herd_0 async  tile (%arg10, %arg11) in (%arg12=%c8_3, %arg13=%c1_2) attributes {id = 3 : i32} {
          %c4_8 = arith.constant 4 : index
          %c8_9 = arith.constant 8 : index
          %c0_10 = arith.constant 0 : index
          %async_token_11, %results_12 = air.execute -> (memref<4x2048xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<4x2048xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<4x2048xbf16, 2 : i32>
          }
          %14 = scf.for %arg14 = %c0_10 to %c8_9 step %c4_8 iter_args(%arg15 = %async_token_11) -> (!air.async.token) {
            %17 = air.channel.get async [%arg15]  @channel_2[%arg10, %arg11] (%results_12[] [] []) {id = 8 : i32} : (memref<4x2048xbf16, 2 : i32>)
            scf.yield %17 : !air.async.token
          }
          %async_token_18 = air.execute [%14] {
            memref.dealloc %results_12 : memref<4x2048xbf16, 2 : i32>
          }
        }
        %async_token_6 = air.execute [%9, %11] {
          memref.dealloc %results : memref<64x2048xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Distribution: the same loop, but a trip takes a single row. The partition's
// two rows are 4 apart in L3, so the L3 put grows a dimension whose stride is
// the original one times the step, and the loop is rewritten to walk the
// partition's rows one at a time. This is the pattern the stride factor exists
// for; the strip-mined case above must not be folded into it.

// CHECK-LABEL: func.func @distributed
// CHECK: air.channel.put{{.*}}@channel_1{{.*}}[%c1{{[_0-9]*}}, %c2{{[_0-9]*}}, %c2048{{[_0-9]*}}] [%c2048{{[_0-9]*}}, %c8192{{[_0-9]*}}, %c1{{[_0-9]*}}]
// CHECK-COUNT-8: memref.alloc() : memref<2x2048xbf16, 1 : i32>
// CHECK: scf.for %{{.*}} = %c0{{[_0-9]*}} to %c2{{[_0-9]*}} step %c1{{[_0-9]*}}

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 * 8 + s1)>
module {
  air.channel @channel_0 []
  air.channel @channel_2 [8, 1]
  func.func @distributed(%arg0: memref<16384x2048xbf16>) {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c256, %arg6=%c1) args(%arg7=%arg0) : memref<16384x2048xbf16> attributes {id = 1 : i32} {
      %1 = affine.apply #map()[%arg3]
      %2 = air.channel.put async  @channel_0[] (%arg7[%1, 0] [64, 2048] [2048, 1]) {id = 1 : i32} : (memref<16384x2048xbf16>)
      %6 = air.segment @seg async  attributes {id = 2 : i32} {
        %c0_0 = arith.constant 0 : index
        %c4_1 = arith.constant 4 : index
        %c1_2 = arith.constant 1 : index
        %c8_3 = arith.constant 8 : index
        %async_token, %results = air.execute -> (memref<64x2048xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x2048xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x2048xbf16, 1 : i32>
        }
        %7 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 4 : i32} : (memref<64x2048xbf16, 1 : i32>)
        %9 = scf.parallel (%arg10) = (%c0_0) to (%c8_3) step (%c1_2) init (%7) -> !air.async.token {
          %14 = scf.for %arg11 = %c0_0 to %c8_3 step %c4_1 iter_args(%arg12 = %7) -> (!air.async.token) {
            %15 = affine.apply #map1()[%arg10, %arg11]
            %16 = air.channel.put async [%arg12]  @channel_2[%arg10, %c0_0] (%results[%15, 0] [1, 2048] [2048, 1]) {id = 5 : i32} : (memref<64x2048xbf16, 1 : i32>)
            scf.yield %16 : !air.async.token
          }
          scf.reduce(%14 : !air.async.token) {
          ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
            %15 = air.wait_all async [%arg11, %arg12]
            scf.reduce.return %15 : !air.async.token
          }
        }
        %11 = air.herd @herd_0 async  tile (%arg10, %arg11) in (%arg12=%c8_3, %arg13=%c1_2) attributes {id = 3 : i32} {
          %c4_8 = arith.constant 4 : index
          %c8_9 = arith.constant 8 : index
          %c0_10 = arith.constant 0 : index
          %async_token_11, %results_12 = air.execute -> (memref<1x2048xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x2048xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<1x2048xbf16, 2 : i32>
          }
          %14 = scf.for %arg14 = %c0_10 to %c8_9 step %c4_8 iter_args(%arg15 = %async_token_11) -> (!air.async.token) {
            %17 = air.channel.get async [%arg15]  @channel_2[%arg10, %arg11] (%results_12[] [] []) {id = 8 : i32} : (memref<1x2048xbf16, 2 : i32>)
            scf.yield %17 : !air.async.token
          }
          %async_token_18 = air.execute [%14] {
            memref.dealloc %results_12 : memref<1x2048xbf16, 2 : i32>
          }
        }
        %async_token_6 = air.execute [%9, %11] {
          memref.dealloc %results : memref<64x2048xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
