//===- opt_memtile_dma_bds.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu1" -split-input-file | FileCheck %s

// RUN: not air-opt %s 2>&1 -air-opt-memtile-dma-bds="device=xcvc1902" -split-input-file | FileCheck %s --check-prefix=AIE1

// Optimize logical air.channel.put/get op into efficient shim dma block descriptor (BD).

// CHECK-LABEL: @func0
// This test exercises pre-fold == maxNumDims with the IV in a NON-outermost
// offset position (offset[1] = %arg4). PR #1658 originally admitted such
// folds, but the post-PR1658 refinement (this PR) only admits when the IV
// is in offset[0] — otherwise the fold either fails to collapse with the
// existing outermost or loses iteration semantics for paired ops (gemm 29
// channel_10-17 regression). Reject path here: the AIR-level unroll
// fallback produces 4 separate puts (one per (%arg0, %arg2) coordinate),
// matching the pre-PR1658 origin/main behavior.
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put async {{.*}} @channel_0[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c0{{.*}}] [%c16{{.*}}, %c4{{.*}}, %c32{{.*}}, %c8{{.*}}] [%c1024{{.*}}, %c8{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: {air.segment_end}

// AIE1: error{{.*}}'func.func' op AIE1 architecture does not come with memtiles.

module {
  air.channel @channel_0 [1, 1]
  func.func @func0() {
    %0 = air.launch async () in () {
      %1 = air.segment @segment_0 async  {
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c16384 = arith.constant 16384 : index
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %async_token, %results = air.execute -> (memref<2x16x32x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<2x16x32x32xbf16, 1>
          air.execute_terminator %alloc : memref<2x16x32x32xbf16, 1>
        }
        %2 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %async_token) -> (!air.async.token) {
          %3 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1) -> (!air.async.token) {
            %4 = scf.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %arg3) -> (!air.async.token) {
              %5 = air.channel.put async [%arg5]  @channel_0[] (%results[%arg0, %arg4, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 21 : i32} : (memref<2x16x32x32xbf16, 1>)
              scf.yield %5 : !air.async.token
            }
            scf.yield %4 : !air.async.token
          }
          scf.yield %3 : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<2x16x32x32xbf16, 1>
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @func1
// CHECK: air.channel.get async {{.*}} @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c1024{{.*}}, %c8192{{.*}}, %c32{{.*}}, %c1{{.*}}])
// CHECK: air.channel.get async {{.*}} @channel_0[%c0{{.*}}, %c1{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c2048{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c1024{{.*}}, %c8192{{.*}}, %c32{{.*}}, %c1{{.*}}])

// AIE1: error{{.*}}'func.func' op AIE1 architecture does not come with memtiles.

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  air.channel @channel_0 [1, 2]
  func.func @func1() {
    %0 = air.launch async () in () {
      %1 = air.segment @segment_0 async  {
        %c2 = arith.constant 2 : index
        %c1024 = arith.constant 1024 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %c2048 = arith.constant 2048 : index
        %async_token, %results = air.execute -> (memref<8x2x32x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<8x2x32x32xbf16, 1>
          air.execute_terminator %alloc : memref<8x2x32x32xbf16, 1>
        }
        %2 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %async_token) -> (!air.async.token) {
          %4 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %arg1) -> (!air.async.token) {
            %5 = air.channel.get async [%arg3]  @channel_0[%c0, %c0] (%results[%arg2, %arg0, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1]) {id = 54 : i32} : (memref<8x2x32x32xbf16, 1>)
            scf.yield %5 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        %3 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %async_token) -> (!air.async.token) {
          %4 = scf.for %arg2 = %c0 to %c8 step %c4 iter_args(%arg3 = %arg1) -> (!air.async.token) {
            %async_token_1, %results_2 = air.execute -> (index) {
              %6 = affine.apply #map()[%arg2]
              air.execute_terminator %6 : index
            }
            %5 = air.channel.get async [%async_token_1, %arg3]  @channel_0[%c0, %c1] (%results[%results_2, %arg0, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1]) {id = 55 : i32} : (memref<8x2x32x32xbf16, 1>)
            scf.yield %5 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<8x2x32x32xbf16, 1>
        }
      }
    }
    return
  }
}

// -----

// =============================================================================
// Regression: scf.for whose channel op already has maxNumDims=4 active wrap
// dims, but the new outer dim folded from the loop can collapse with an
// existing adjacent dim (stride[i-1] == size[i] * stride[i]) under
// canonicalization. The pre-fold gate must NOT pre-reject these; the
// post-canonicalize legality check decides.
//
// Concretely: existing wrap <2, 6272> · <98, 8> · <8, 784> · <8, 1> (12544 B)
// inside scf.for %xb = 0..4 step 1 with offsets[1] = %xb at stride 12544.
// Fold prepends <4, 12544>; canonicalize collapses <4, 12544> · <2, 6272>
// → <8, 6272>; final 4D <8, 6272> · <98, 8> · <8, 784> · <8, 1> (50176 B).
// (This is the L2->L1 act gather pattern from conv2d_14x14.)
// =============================================================================

// CHECK-LABEL: @postfold_collapse_4d_to_4d
// CHECK-NOT:   scf.for
// CHECK:       air.channel.put async {{.*}} @channel_pf[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c8{{.*}}, %c98{{.*}}, %c8{{.*}}, %c8{{.*}}] [%c6272{{.*}}, %c8{{.*}}, %c784{{.*}}, %c1{{.*}}])
// CHECK-NOT:   air.channel.put
// CHECK:       {air.segment_end}

module {
  air.channel @channel_pf [1]
  func.func @postfold_collapse_4d_to_4d() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg_pf async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c98 = arith.constant 98 : index
        %c784 = arith.constant 784 : index
        %c6272 = arith.constant 6272 : index
        %c12544 = arith.constant 12544 : index
        %c50176 = arith.constant 50176 : index
        %async_token, %results = air.execute -> (memref<1x3x50176xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x3x50176xi8, 1 : i32>
          air.execute_terminator %alloc : memref<1x3x50176xi8, 1 : i32>
        }
        %t = scf.for %xb = %c0 to %c4 step %c1 iter_args(%arg = %async_token) -> (!air.async.token) {
          %p = air.channel.put async [%arg] @channel_pf[] (%results[%c0, %xb, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c98, %c8, %c8] [%c50176, %c12544, %c6272, %c8, %c784, %c1]) {id = 1 : i32} : (memref<1x3x50176xi8, 1 : i32>)
          scf.yield %p : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<1x3x50176xi8, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// =============================================================================
// Regression: scf.for whose channel op has maxNumDims=4 active wrap dims AND
// the new outer dim folded from the loop CANNOT collapse with any existing
// adjacent dim. The post-canonicalize legality check must reject the fold to
// avoid emitting an illegal 5D BD wrap; the unroll fallback then materializes
// N separate channel ops.
//
// Concretely: existing wrap <2, 200> · <3, 50> · <4, 12> · <5, 1> (no
// internal collapsing — stride[i-1] != size[i] * stride[i] for any i), loop
// adds outer with stride 200 (matches stride[0] but 200 != 2 * 200, so no
// collapse). Result IR has 3 separate puts (unroll fallback), each with
// the original 4D wrap.
// =============================================================================

// CHECK-LABEL: @postfold_no_collapse_rejected
// CHECK-NOT:   scf.for
// CHECK:       air.channel.put async {{.*}} @channel_neg[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK:       air.channel.put async {{.*}} @channel_neg[] (%{{.*}}[%c1{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK:       air.channel.put async {{.*}} @channel_neg[] (%{{.*}}[%c2{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK-NOT:   [%c{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}]
// CHECK:       {air.segment_end}

module {
  air.channel @channel_neg [1]
  func.func @postfold_no_collapse_rejected() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg_neg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c12 = arith.constant 12 : index
        %c50 = arith.constant 50 : index
        %c200 = arith.constant 200 : index
        %async_token, %results = air.execute -> (memref<10000xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<10000xi8, 1 : i32>
          air.execute_terminator %alloc : memref<10000xi8, 1 : i32>
        }
        %t = scf.for %iv = %c0 to %c3 step %c1 iter_args(%arg = %async_token) -> (!air.async.token) {
          %p = air.channel.put async [%arg] @channel_neg[] (%results[%iv, %c0, %c0, %c0] [%c2, %c3, %c4, %c5] [%c200, %c50, %c12, %c1]) {id = 2 : i32} : (memref<10000xi8, 1 : i32>)
          scf.yield %p : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<10000xi8, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Cascade-redundant unroll guard: 9 * 16 = 144 > kRedundantUnrollLockLimit
// (16) with all-constant channel offsets and no IV reach. Both loops must
// be preserved so downstream emits one BD with implicit per-iter repeat
// at init=1, instead of 144 chained identical puts at init=144.

// CHECK-LABEL: @unroll_cascade_redundant_preserved
// CHECK:       scf.for {{.*}} = %c0{{.*}} to %c9{{.*}} step %c1{{.*}}
// CHECK:         scf.for {{.*}} = %c0{{.*}} to %c16{{.*}} step %c1{{.*}}
// CHECK:           air.channel.put async {{.*}} @channel_casc[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c8{{.*}}, %c98{{.*}}, %c8{{.*}}, %c8{{.*}}] [%c6272{{.*}}, %c8{{.*}}, %c784{{.*}}, %c1{{.*}}])
// CHECK-NOT:       air.channel.put
// CHECK:         scf.yield
// CHECK:       scf.yield
// CHECK:       {air.segment_end}

module {
  air.channel @channel_casc [1]
  func.func @unroll_cascade_redundant_preserved() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg_casc async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c9 = arith.constant 9 : index
        %c16 = arith.constant 16 : index
        %c98 = arith.constant 98 : index
        %c784 = arith.constant 784 : index
        %c6272 = arith.constant 6272 : index
        %async_token, %results = air.execute -> (memref<1x3x50176xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x3x50176xi8, 1 : i32>
          air.execute_terminator %alloc : memref<1x3x50176xi8, 1 : i32>
        }
        %tg = scf.for %g = %c0 to %c9 step %c1 iter_args(%arg_g = %async_token) -> (!air.async.token) {
          %ty = scf.for %y = %c0 to %c16 step %c1 iter_args(%arg_y = %arg_g) -> (!air.async.token) {
            %p = air.channel.put async [%arg_y] @channel_casc[] (%results[%c0, %c0, %c0, %c0] [%c8, %c98, %c8, %c8] [%c6272, %c8, %c784, %c1]) {id = 3 : i32} : (memref<1x3x50176xi8, 1 : i32>)
            scf.yield %p : !air.async.token
          }
          scf.yield %ty : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<1x3x50176xi8, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Regression guard: short redundant loop (trip 4 <= 16) still unrolls.
// Keeps the cascade guard from over-rejecting workloads that need small
// no-IV-in-offsets unrolls.

// CHECK-LABEL: @unroll_small_redundant_admitted
// CHECK-NOT:   scf.for
// CHECK:       air.channel.put async {{.*}} @channel_small[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK:       air.channel.put async {{.*}} @channel_small[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK:       air.channel.put async {{.*}} @channel_small[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK:       air.channel.put async {{.*}} @channel_small[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c4{{.*}}, %c5{{.*}}] [%c200{{.*}}, %c50{{.*}}, %c12{{.*}}, %c1{{.*}}])
// CHECK:       {air.segment_end}

module {
  air.channel @channel_small [1]
  func.func @unroll_small_redundant_admitted() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg_small async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c5 = arith.constant 5 : index
        %c12 = arith.constant 12 : index
        %c50 = arith.constant 50 : index
        %c200 = arith.constant 200 : index
        %async_token, %results = air.execute -> (memref<10000xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<10000xi8, 1 : i32>
          air.execute_terminator %alloc : memref<10000xi8, 1 : i32>
        }
        %t = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg = %async_token) -> (!air.async.token) {
          %p = air.channel.put async [%arg] @channel_small[] (%results[%c0, %c0, %c0, %c0] [%c2, %c3, %c4, %c5] [%c200, %c50, %c12, %c1]) {id = 4 : i32} : (memref<10000xi8, 1 : i32>)
          scf.yield %p : !air.async.token
        }
        %async_token_0 = air.execute {
          memref.dealloc %results : memref<10000xi8, 1 : i32>
        }
      }
    }
    return
  }
}
