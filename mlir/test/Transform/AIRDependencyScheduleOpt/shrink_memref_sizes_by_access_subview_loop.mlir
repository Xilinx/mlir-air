//===- shrink_memref_sizes_by_access_subview_loop.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file | FileCheck %s

// L1 buffers reached only through memref.subview inside an scf.for whose IV
// drives the subview offset. The loop counts toward the buffer's extent even
// though every user sits inside it: the shrink keeps the loop term in the
// offset, so a buffer sized for one iteration is indexed out of bounds by the
// next.

// Four 8-row slices at rows 0, 8, 16, 24 cover 32 rows.

// CHECK-LABEL: func.func @strided_slices
// CHECK: memref.alloc() {air.shrinkage = true} : memref<32x8xf32, 2>
module {
  func.func @strided_slices() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<64x8xf32, 2>) {
          %a = memref.alloc() : memref<64x8xf32, 2>
          air.execute_terminator %a : memref<64x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<64x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %f0 = arith.constant 0.000000e+00 : f32
          %step = arith.constant 8 : index
          %ub = arith.constant 32 : index
          scf.for %i = %c0 to %ub step %step {
            %sv = memref.subview %buf[%i, 0] [8, 8] [1, 1] : memref<64x8xf32, 2> to memref<8x8xf32, strided<[8, 1], offset: ?>, 2>
            linalg.fill ins(%f0 : f32) outs(%sv : memref<8x8xf32, strided<[8, 1], offset: ?>, 2>)
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<64x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// Overlapping 8-row windows at rows 0, 1, 2, 3 cover (4 - 1) * 1 + 8 = 11
// rows, not one per trip.

// CHECK-LABEL: func.func @overlapping_windows
// CHECK: memref.alloc() {air.shrinkage = true} : memref<11x8xf32, 2>
module {
  func.func @overlapping_windows() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<64x8xf32, 2>) {
          %a = memref.alloc() : memref<64x8xf32, 2>
          air.execute_terminator %a : memref<64x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<64x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %f0 = arith.constant 0.000000e+00 : f32
          %step = arith.constant 1 : index
          %ub = arith.constant 4 : index
          scf.for %i = %c0 to %ub step %step {
            %sv = memref.subview %buf[%i, 0] [8, 8] [1, 1] : memref<64x8xf32, 2> to memref<8x8xf32, strided<[8, 1], offset: ?>, 2>
            linalg.fill ins(%f0 : f32) outs(%sv : memref<8x8xf32, strided<[8, 1], offset: ?>, 2>)
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<64x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// A rank-reducing subview that drops a middle unit dimension: its result
// layout keeps the strides of dims 0, 1 and 3. Each dimension's stride comes
// from the source layout, so the loop's range lands on dim 2 and the herd
// dimensions shrink to 1.

// CHECK-LABEL: func.func @rank_reduced_middle_dim
// CHECK: memref.alloc() {air.shrinkage = true} : memref<1x1x4x4xf32, 2>
module {
  func.func @rank_reduced_middle_dim() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<2x2x4x4xf32, 2>) {
          %a = memref.alloc() : memref<2x2x4x4xf32, 2>
          air.execute_terminator %a : memref<2x2x4x4xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<2x2x4x4xf32, 2> {
          %c0 = arith.constant 0 : index
          %f0 = arith.constant 0.000000e+00 : f32
          %step = arith.constant 1 : index
          %ub = arith.constant 4 : index
          scf.for %i = %c0 to %ub step %step {
            %sv = memref.subview %buf[%tx, %ty, %i, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<2x2x4x4xf32, 2> to memref<1x1x4xf32, strided<[32, 16, 1], offset: ?>, 2>
            linalg.fill ins(%f0 : f32) outs(%sv : memref<1x1x4xf32, strided<[32, 16, 1], offset: ?>, 2>)
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<2x2x4x4xf32, 2>
        }
      }
    }
    return
  }
}
