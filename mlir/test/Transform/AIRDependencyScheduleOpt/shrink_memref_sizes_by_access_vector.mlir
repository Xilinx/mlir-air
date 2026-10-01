//===- shrink_memref_sizes_by_access_vector.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file | FileCheck %s

// L1 buffers reached only by vector transfers. Each dimension is sized to the
// highest index the transfer starts at plus the vector's extent along it. The
// shrink leaves loop terms and constants in the index, so the bound counts
// from 0, not from where the accesses start.

// A loop from 8 to 32 step 8: rows 8, 16 and 24, so 25 rows.

// CHECK-LABEL: func.func @loop_lower_bound
// CHECK: memref.alloc() {air.shrinkage = true} : memref<25x8xf32, 2>
module {
  func.func @loop_lower_bound() {
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
          %c8 = arith.constant 8 : index
          %c32 = arith.constant 32 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
          scf.for %i = %c8 to %c32 step %c8 {
            vector.transfer_write %cst, %buf[%i, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<64x8xf32, 2>
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

// A constant index: row 8, so 9 rows.

// CHECK-LABEL: func.func @constant_offset
// CHECK: memref.alloc() {air.shrinkage = true} : memref<9x8xf32, 2>
module {
  func.func @constant_offset() {
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
          %c8 = arith.constant 8 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
          vector.transfer_write %cst, %buf[%c8, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<64x8xf32, 2>
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

// A transposed transfer: vector dimension 0 runs along memref dimension 1, so
// a vector<8x4> covers 4 rows and 8 columns.

// CHECK-LABEL: func.func @transposed
// CHECK: memref.alloc() {air.shrinkage = true} : memref<4x8xf32, 2>
module {
  func.func @transposed() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<16x64xf32, 2>) {
          %a = memref.alloc() : memref<16x64xf32, 2>
          air.execute_terminator %a : memref<16x64xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<16x64xf32, 2> {
          %c0 = arith.constant 0 : index
          %cst = arith.constant dense<0.000000e+00> : vector<8x4xf32>
          vector.transfer_write %cst, %buf[%c0, %c0] {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<8x4xf32>, memref<16x64xf32, 2>
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<16x64xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// A read and a write at constant rows 6 and 2: the bound covers both, 7 rows.

// CHECK-LABEL: func.func @read_and_write
// CHECK: memref.alloc() {air.shrinkage = true} : memref<7x8xf32, 2>
module {
  func.func @read_and_write() {
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
          %c6 = arith.constant 6 : index
          %c2_0 = arith.constant 2 : index
          %f0 = arith.constant 0.000000e+00 : f32
          %r = vector.transfer_read %buf[%c6, %c0], %f0 {in_bounds = [true, true]} : memref<64x8xf32, 2>, vector<1x8xf32>
          vector.transfer_write %r, %buf[%c2_0, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<64x8xf32, 2>
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

// A broadcast read: vector dimension 0 repeats one memref row, so the read at
// row 5 covers row 5 alone. With the write at row 2, 6 rows.

// CHECK-LABEL: func.func @broadcast_read
// CHECK: memref.alloc() {air.shrinkage = true} : memref<6x8xf32, 2>
module {
  func.func @broadcast_read() {
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
          %c2_0 = arith.constant 2 : index
          %c5 = arith.constant 5 : index
          %f0 = arith.constant 0.000000e+00 : f32
          %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
          vector.transfer_write %cst, %buf[%c2_0, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<64x8xf32, 2>
          %r = vector.transfer_read %buf[%c5, %c0], %f0 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (0, d1)>} : memref<64x8xf32, 2>, vector<4x8xf32>
          vector.transfer_write %r, %buf[%c2_0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<64x8xf32, 2>
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

// An index from affine.delinearize_index of a loop over 0 to 7 by (4, 2): the
// results reach 3 and 1, so 4 and 2 along those dimensions.

// CHECK-LABEL: func.func @delinearized_index
// CHECK: memref.alloc() {air.shrinkage = true} : memref<4x2x8xf32, 2>
module {
  func.func @delinearized_index() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<8x4x8xf32, 2>) {
          %a = memref.alloc() : memref<8x4x8xf32, 2>
          air.execute_terminator %a : memref<8x4x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<8x4x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8xf32>
          scf.for %i = %c0 to %c8 step %c1_0 {
            %t, %d:2 = air.execute -> (index, index) {
              %q:2 = affine.delinearize_index %i into (4, 2) : index, index
              air.execute_terminator %q#0, %q#1 : index, index
            }
            vector.transfer_write %cst, %buf[%d#0, %d#1, %c0] {in_bounds = [true, true, true]} : vector<1x1x8xf32>, memref<8x4x8xf32, 2>
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<8x4x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// An index computed by an op the shrink does not model (here arith.divui)
// keeps its dimension whole; the other dimension still shrinks to the vector's
// 8 columns.

// CHECK-LABEL: func.func @unbounded_index
// CHECK: memref.alloc() {air.shrinkage = true} : memref<16x8xf32, 2>
module {
  func.func @unbounded_index() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<16x16xf32, 2>) {
          %a = memref.alloc() : memref<16x16xf32, 2>
          air.execute_terminator %a : memref<16x16xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<16x16xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
          scf.for %i = %c0 to %c4 step %c1_0 {
            %n = arith.divui %i, %c2_0 : index
            vector.transfer_write %cst, %buf[%n, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<16x16xf32, 2>
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<16x16xf32, 2>
        }
      }
    }
    return
  }
}
