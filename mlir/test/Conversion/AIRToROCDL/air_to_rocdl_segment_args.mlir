//===- air_to_rocdl_segment_args.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s

// air.segment block args are laid out [ids..., sizes..., kernel_args...], so
// when a segment declares an iteration space its kernel args do not start at
// block-arg 0. Flattening has to index past the ids and sizes; open-coding the
// offset makes the pass read past the end of the operand list and abort.
//
// The segment here is 2-D, so %s0/%s1/%s2 are block args 4/5/6. The space is
// all ones, which is the only shape -air-to-rocdl can flatten (gpu.launch's
// grid belongs to air.launch and its block to air.herd), so %sx lowers to the
// constant 0 and the body runs once.
//
// Memrefs are #air.symmetric_heap so the herd may address them directly: the
// per-level NPU rule in verifyComputeMemoryAccess is waived for the GPU
// symmetric heap.

// CHECK-LABEL: func.func @segment_iteration_space
// CHECK-SAME:    %[[A:[^:]*]]: memref<64xf32, #air.symmetric_heap>
// CHECK-SAME:    %[[B:[^:]*]]: memref<64xf32, #air.symmetric_heap>
// CHECK-SAME:    %[[C:[^:]*]]: memref<64xf32, #air.symmetric_heap>
// CHECK:       gpu.launch
// The segment id collapses to 0, and the kernel args name the func arguments.
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         memref.load %[[A]][%[[ZERO]]]
// CHECK:         memref.load %[[B]]
// CHECK:         memref.store %{{.*}}, %[[C]]
// CHECK:         gpu.terminator
// CHECK-NOT:   air.segment
// CHECK-NOT:   air.herd
// CHECK-NOT:   air.launch

module {
  func.func @segment_iteration_space(%arg0: memref<64xf32, #air.symmetric_heap>,
                                     %arg1: memref<64xf32, #air.symmetric_heap>,
                                     %arg2: memref<64xf32, #air.symmetric_heap>) {
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%in0=%arg0, %in1=%arg1, %out=%arg2) : memref<64xf32, #air.symmetric_heap>, memref<64xf32, #air.symmetric_heap>, memref<64xf32, #air.symmetric_heap> {
      // air.launch is IsolatedFromAbove: the segment's size operands must be
      // defined inside the launch body, not at function scope.
      %c1_l = arith.constant 1 : index
      air.segment @seg unroll (%sx, %sy) in (%nsx=%c1_l, %nsy=%c1_l) args(%s0=%in0, %s1=%in1, %s2=%out) : memref<64xf32, #air.symmetric_heap>, memref<64xf32, #air.symmetric_heap>, memref<64xf32, #air.symmetric_heap> {
        %c4_s = arith.constant 4 : index
        %c1_s = arith.constant 1 : index
        // %sx is block arg 0, %s0 is block arg 4: the two must not be confused.
        air.herd @herd tile (%tx, %ty) in (%ntx=%c4_s, %nty=%c1_s) args(%h0=%s0, %h1=%s1, %h2=%s2, %hid=%sx) : memref<64xf32, #air.symmetric_heap>, memref<64xf32, #air.symmetric_heap>, memref<64xf32, #air.symmetric_heap>, index {
          %a = memref.load %h0[%hid] : memref<64xf32, #air.symmetric_heap>
          %b = memref.load %h1[%hid] : memref<64xf32, #air.symmetric_heap>
          %c = arith.addf %a, %b : f32
          memref.store %c, %h2[%hid] : memref<64xf32, #air.symmetric_heap>
        }
      }
    }
    return
  }
}
