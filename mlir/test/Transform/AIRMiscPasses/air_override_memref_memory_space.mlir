//===- air_override_memref_memory_space.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-override-memref-memory-space="scope=herd memory-space=2" | FileCheck %s
// RUN: air-opt %s -air-override-memref-memory-space="scope=launch memory-space=2" | FileCheck %s --check-prefix=LAUNCH
// RUN: air-opt %s -air-override-memref-memory-space="scope=segment memory-space=1" | FileCheck %s --check-prefix=SEGMENT
// RUN: air-opt %s -air-override-memref-memory-space="scope=func memory-space=1" | FileCheck %s --check-prefix=FUNC

module {

  // CHECK-LABEL: func.func @func0
  // CHECK: memref.alloc() : memref<32x64xf32, 2 : i32>
  // LAUNCH-LABEL: func.func @func0
  // scope=launch is exclusive: alloc inside herd/segment is unchanged
  // LAUNCH: memref.alloc() : memref<32x64xf32, 3>
  // MS1-LABEL: func.func @func0
  // MS1: memref.alloc() : memref<32x64xf32, 2 : i32>

  func.func @func0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg9, %arg10) in (%arg11=%c1, %arg12=%c1_0) args(%arg13=%arg0, %arg14=%arg1, %arg15=%arg2) : memref<*xf32>, memref<*xf32>, memref<*xf32> {
      air.segment @bare_matmul_0  args(%arg16=%arg9, %arg17=%arg10, %arg18=%arg11, %arg19=%arg12, %arg20=%arg13, %arg21=%arg14, %arg22=%arg15) : index, index, index, index, memref<*xf32>, memref<*xf32>, memref<*xf32> {
        %c2_1 = arith.constant 2 : index
        %c2_2 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg23, %arg24) in (%arg25=%c2_1, %arg26=%c2_2) args(%arg27=%arg16, %arg28=%arg17, %arg29=%arg18, %arg30=%arg19, %arg31=%arg20, %arg32=%arg21, %arg33=%arg22) : index, index, index, index, memref<*xf32>, memref<*xf32>, memref<*xf32> {
          %c32 = arith.constant 32 : index
          %c2048 = arith.constant 2048 : index
          %c64 = arith.constant 64 : index
          %1 = arith.muli %arg23, %c2048 : index
          %reinterpret_cast = memref.reinterpret_cast %arg31 to offset: [%1], sizes: [32, 64], strides: [%c64, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
          %alloc = memref.alloc() : memref<32x64xf32, 3>
          memref.copy %reinterpret_cast, %alloc : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32, 3>
        }
      }
    }
    return
  }

  // LAUNCH-LABEL: func.func @func1
  // LAUNCH: memref.alloc() : memref<8x4x4x8xf32, 2 : i32>
  // LAUNCH: memref.collapse_shape {{.*}} : memref<8x4x4x8xf32, 2 : i32> into memref<32x32xf32, 2 : i32>
  // LAUNCH: memref.alloc() : memref<4x8x8x4xf32, 2 : i32>
  // LAUNCH: memref.collapse_shape {{.*}} : memref<4x8x8x4xf32, 2 : i32> into memref<32x32xf32, 2 : i32>
  // LAUNCH: memref.alloc() : memref<32x32xf32, 2 : i32>
  // LAUNCH: memref.expand_shape {{.*}} : memref<32x32xf32, 2 : i32> into memref<8x4x8x4xf32, 2 : i32>
  // LAUNCH: memref.alloc() : memref<8x8x4x4xf32, 2 : i32>
  // MS1-LABEL: func.func @func1
  // MS1: memref.alloc() : memref<8x4x4x8xf32, 1 : i32>
  // MS1: memref.collapse_shape {{.*}} : memref<8x4x4x8xf32, 1 : i32> into memref<32x32xf32, 1 : i32>
  // MS1: memref.alloc() : memref<4x8x8x4xf32, 1 : i32>
  // MS1: memref.collapse_shape {{.*}} : memref<4x8x8x4xf32, 1 : i32> into memref<32x32xf32, 1 : i32>
  // MS1: memref.alloc() : memref<32x32xf32, 1 : i32>
  // MS1: memref.expand_shape {{.*}} : memref<32x32xf32, 1 : i32> into memref<8x4x8x4xf32, 1 : i32>
  // MS1: memref.alloc() : memref<8x8x4x4xf32, 1 : i32>

  func.func @func1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c2, %arg13=%c2, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xf32>, memref<*xf32>, memref<*xf32> {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<8x4x4x8xf32>
      %collapse_shape = memref.collapse_shape %alloc_2 [[0, 1], [2, 3]] : memref<8x4x4x8xf32> into memref<32x32xf32>
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x8x8x4xf32>
      %collapse_shape_4 = memref.collapse_shape %alloc_3 [[0, 1], [2, 3]] : memref<4x8x8x4xf32> into memref<32x32xf32>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      linalg.matmul ins(%collapse_shape, %collapse_shape_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
      %expand_shape = memref.expand_shape %alloc_5 [[0, 1], [2, 3]] output_shape [8, 4, 8, 4] : memref<32x32xf32> into memref<8x4x8x4xf32>
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8x8x4x4xf32>
      linalg.transpose ins(%expand_shape : memref<8x4x8x4xf32>) outs(%alloc_6 : memref<8x8x4x4xf32>) permutation = [0, 2, 1, 3]
    }
    return
  }

  // Test exclusive scoping: scope=herd should only change herd allocs,
  // scope=segment should only change segment allocs (issue #1379).

  // CHECK-LABEL: func.func @func_exclusive_scope
  // scope=herd: herd alloc changes to memory_space 2, segment alloc unchanged
  // CHECK: air.segment
  // CHECK:   memref.alloc() : memref<64xf32, 3>
  // CHECK:   air.herd
  // CHECK:     memref.alloc() : memref<32xf32, 2 : i32>

  // SEGMENT-LABEL: func.func @func_exclusive_scope
  // scope=segment: segment alloc changes to memory_space 1, herd alloc unchanged
  // SEGMENT: air.segment
  // SEGMENT:   memref.alloc() : memref<64xf32, 1 : i32>
  // SEGMENT:   air.herd
  // SEGMENT:     memref.alloc() : memref<32xf32, 3>

  func.func @func_exclusive_scope(%arg0: memref<64xf32, 3>, %arg1: memref<32xf32, 3>) {
    air.launch () in () args(%a0=%arg0, %a1=%arg1) : memref<64xf32, 3>, memref<32xf32, 3> {
      air.segment @seg args(%s0=%a0, %s1=%a1) : memref<64xf32, 3>, memref<32xf32, 3> {
        %c1 = arith.constant 1 : index
        %seg_buf = memref.alloc() : memref<64xf32, 3>
        memref.copy %s0, %seg_buf : memref<64xf32, 3> to memref<64xf32, 3>
        air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h1=%s1) : memref<32xf32, 3> {
          %herd_buf = memref.alloc() : memref<32xf32, 3>
          memref.copy %h1, %herd_buf : memref<32xf32, 3> to memref<32xf32, 3>
        }
      }
    }
    return
  }

  // Test scope=func with herd directly inside func (no launch/segment).
  // scope=func should NOT override the herd alloc (issue #1379 follow-up).

  // FUNC-LABEL: func.func @func_herd_no_launch
  // scope=func changes func-level alloc from 3 to 1, herd alloc stays at 3
  // FUNC: memref.alloc() : memref<64xf32, 1 : i32>
  // FUNC: air.herd
  // FUNC:   memref.alloc() : memref<32xf32, 3>

  func.func @func_herd_no_launch(%arg0: memref<32xf32, 3>) {
    %c1 = arith.constant 1 : index
    // Alloc at func level (outside herd) — starts at memory_space 3
    %func_buf = memref.alloc() : memref<64xf32, 3>
    memref.dealloc %func_buf : memref<64xf32, 3>
    air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h0=%arg0) : memref<32xf32, 3> {
      // Alloc at herd level — should NOT be changed by scope=func
      %herd_buf = memref.alloc() : memref<32xf32, 3>
      memref.copy %h0, %herd_buf : memref<32xf32, 3> to memref<32xf32, 3>
    }
    return
  }

  // Test that subview source type is updated when herd arg type changes
  // due to segment-level alloc override (issue #1384).

  // SEGMENT-LABEL: func.func @func_subview_type_propagation
  // SEGMENT: air.segment
  // SEGMENT:   memref.alloc() : memref<64xbf16, 1 : i32>
  // SEGMENT:   air.herd
  // SEGMENT:     memref.subview %{{.*}}[0] [32] [1] : memref<64xbf16, 1 : i32> to memref<32xbf16, strided<[1]>, 1 : i32>

  func.func @func_subview_type_propagation() {
    air.launch () in () {
      air.segment @seg {
        %c1 = arith.constant 1 : index
        %buf = memref.alloc() : memref<64xbf16, 3>
        air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h0=%buf) : memref<64xbf16, 3> {
          %local = memref.alloc() : memref<32xbf16, 3>
          %sv = memref.subview %h0[0] [32] [1] : memref<64xbf16, 3> to memref<32xbf16, strided<[1]>, 3>
          memref.copy %sv, %local : memref<32xbf16, strided<[1]>, 3> to memref<32xbf16, 3>
          memref.dealloc %local : memref<32xbf16, 3>
        }
        memref.dealloc %buf : memref<64xbf16, 3>
      }
    }
    return
  }

  // Test that allocs with explicit non-default memory space are preserved
  // while unset memory space allocs are overridden (PR #1436).
  // Uses memory_space=3 (raw int) for "unassigned" allocs to satisfy verifier,
  // and memory_space=2 (L1) / memory_space=1 (L2) for "already set" allocs.

  // CHECK-LABEL: func.func @func_skip_existing_memspace
  // scope=herd, target=L1(2): unassigned herd alloc overridden, L1 alloc preserved
  // CHECK: air.herd
  // CHECK:   memref.alloc() : memref<32xf32, 2 : i32>
  // CHECK:   memref.alloc() : memref<32xf32, 2 : i32>

  // SEGMENT-LABEL: func.func @func_skip_existing_memspace
  // scope=segment, target=L2(1): unassigned segment alloc overridden,
  // L2 segment alloc preserved, herd allocs unchanged by scope exclusion
  // SEGMENT: air.segment
  // SEGMENT:   memref.alloc() : memref<64xf32, 1 : i32>
  // SEGMENT:   memref.alloc() : memref<64xf32, 1 : i32>
  // SEGMENT:   air.herd
  // SEGMENT:     memref.alloc() : memref<32xf32, 3>
  // SEGMENT:     memref.alloc() : memref<32xf32, 2 : i32>

  func.func @func_skip_existing_memspace(%arg0: memref<*xf32>) {
    air.launch () in () args(%a0=%arg0) : memref<*xf32> {
      air.segment @seg args(%s0=%a0) : memref<*xf32> {
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %rc64 = memref.reinterpret_cast %s0 to offset: [0], sizes: [64], strides: [1] : memref<*xf32> to memref<64xf32, strided<[1], offset: ?>>
        // Unassigned (memory_space=3) alloc at segment level - should be overridden
        %seg_default = memref.alloc() : memref<64xf32, 3>
        memref.copy %rc64, %seg_default : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 3>
        // Explicit L2 (memory_space=1) alloc at segment level - should be preserved
        %seg_l2 = memref.alloc() : memref<64xf32, 1 : i32>
        memref.copy %rc64, %seg_l2 : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, 1 : i32>
        air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%h0=%s0) : memref<*xf32> {
          %c32 = arith.constant 32 : index
          %rc32 = memref.reinterpret_cast %h0 to offset: [0], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
          // Unassigned (memory_space=3) alloc at herd level - should be overridden by scope=herd
          %herd_default = memref.alloc() : memref<32xf32, 3>
          memref.copy %rc32, %herd_default : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32, 3>
          // Explicit L1 (memory_space=2) alloc at herd level - should be preserved
          %herd_l1 = memref.alloc() : memref<32xf32, 2 : i32>
          memref.copy %rc32, %herd_l1 : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32, 2 : i32>
        }
      }
    }
    return
  }
}
