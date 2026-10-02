//===- shrink_memref_sizes_by_access_affine_apply.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file | FileCheck %s

// L1 buffers reached only by vector transfers inside a 2x2 herd, at indices
// that combine loop IVs with a herd tile index. Each dimension is sized to the
// highest index it reaches once the shrink has rewritten it, plus the vector's
// extent along it. A tile index the rewrite removes counts as 0; one it cannot
// remove counts over every tile. An index that cannot be bounded keeps the
// dimension's full size.

// A bare affine.apply, coefficient 1 on the IV: each tile walks a 2x2 block
// of 8x8 tiles, and the tile term is dropped from the rewritten indices.

// CHECK-LABEL: func.func @apply_coef1
// CHECK: memref.alloc() {air.shrinkage = true} : memref<2x2x8x8xf32, 2>
// CHECK: air.herd
// CHECK: %[[N:.*]] = affine.apply #{{.*}}()[%{{.*}}, %c0]
// CHECK: %[[M:.*]] = affine.apply #{{.*}}()[%{{.*}}, %c0]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[N]], %[[M]], %c0{{.*}}, %c0{{.*}}] {{.*}} : vector<1x1x8x8xf32>, memref<2x2x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
module {
  func.func @apply_coef1() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<4x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<4x4x8x8xf32, 2>
          air.execute_terminator %a : memref<4x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<4x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          scf.for %i = %c0 to %c2_0 step %c1_0 {
            scf.for %j = %c0 to %c2_0 step %c1_0 {
              %n = affine.apply #map()[%i, %ty]
              %m = affine.apply #map()[%j, %tx]
              vector.transfer_write %cst, %buf[%n, %m, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<4x4x8x8xf32, 2>
            }
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<4x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// A read-modify-write, as an accumulator is updated: the transfer_read is
// sized the same way as the write.

// CHECK-LABEL: func.func @apply_read_write
// CHECK: memref.alloc() {air.shrinkage = true} : memref<2x2x8x8xf32, 2>
// CHECK: vector.transfer_read {{.*}} : memref<2x2x8x8xf32, 2>, vector<1x1x8x8xf32>
// CHECK: vector.transfer_write {{.*}} : vector<1x1x8x8xf32>, memref<2x2x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
module {
  func.func @apply_read_write() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<4x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<4x4x8x8xf32, 2>
          air.execute_terminator %a : memref<4x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<4x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<1.000000e+00> : vector<1x1x8x8xf32>
          %f0 = arith.constant 0.000000e+00 : f32
          scf.for %i = %c0 to %c2_0 step %c1_0 {
            scf.for %j = %c0 to %c2_0 step %c1_0 {
              %n = affine.apply #map()[%i, %ty]
              %m = affine.apply #map()[%j, %tx]
              %r = vector.transfer_read %buf[%n, %m, %c0, %c0], %f0 {in_bounds = [true, true, true, true]} : memref<4x4x8x8xf32, 2>, vector<1x1x8x8xf32>
              %s = arith.addf %r, %cst : vector<1x1x8x8xf32>
              vector.transfer_write %s, %buf[%n, %m, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<4x4x8x8xf32, 2>
            }
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<4x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// Coefficient 2 on the IV: each tile writes rows 0 and 2, so 3 rows.

// CHECK-LABEL: func.func @apply_coef2
// CHECK: memref.alloc() {air.shrinkage = true} : memref<3x1x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 4)>
module {
  func.func @apply_coef2() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<8x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<8x4x8x8xf32, 2>
          air.execute_terminator %a : memref<8x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<8x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          scf.for %i = %c0 to %c2_0 step %c1_0 {
            %n = affine.apply #map()[%i, %ty]
            vector.transfer_write %cst, %buf[%n, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<8x4x8x8xf32, 2>
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<8x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// The same index computed inside an air.execute.

// CHECK-LABEL: func.func @execute_apply_coef2
// CHECK: memref.alloc() {air.shrinkage = true} : memref<3x1x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 * 2 + s1 * 4)>
module {
  func.func @execute_apply_coef2() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<8x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<8x4x8x8xf32, 2>
          air.execute_terminator %a : memref<8x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<8x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          scf.for %i = %c0 to %c2_0 step %c1_0 {
            %tn, %n = air.execute -> (index) {
              %v = affine.apply #map()[%i, %ty]
              air.execute_terminator %v : index
            }
            vector.transfer_write %cst, %buf[%n, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<8x4x8x8xf32, 2>
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<8x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// Not linear in the IV, so no single stride: that dimension keeps its full
// size, while the constant-indexed one still shrinks.

// CHECK-LABEL: func.func @iv_floordiv
// CHECK: memref.alloc() {air.shrinkage = true} : memref<20x1x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 + s0 floordiv 4 + s1 * 10)>
module {
  func.func @iv_floordiv() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<20x2x8x8xf32, 2>) {
          %a = memref.alloc() : memref<20x2x8x8xf32, 2>
          air.execute_terminator %a : memref<20x2x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<20x2x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          %c8 = arith.constant 8 : index
          scf.for %i = %c0 to %c8 step %c1_0 {
            %n = affine.apply #map()[%i, %tx]
            vector.transfer_write %cst, %buf[%n, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<20x2x8x8xf32, 2>
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<20x2x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// Two loops drive one index, i + 4 * j for i < 4 and j < 8: rows 0 to 31 per
// tile once the tile term is removed.

// CHECK-LABEL: func.func @two_ivs
// CHECK: memref.alloc() {air.shrinkage = true} : memref<32x1x8x8xf32, 2>
// CHECK: affine.apply #{{.*}}()[%{{.*}}, %{{.*}}, %c0]
#map = affine_map<()[s0, s1, s2] -> (s0 + s1 * 4 + s2 * 32)>
module {
  func.func @two_ivs() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<64x2x8x8xf32, 2>) {
          %a = memref.alloc() : memref<64x2x8x8xf32, 2>
          air.execute_terminator %a : memref<64x2x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<64x2x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          %c4 = arith.constant 4 : index
          %c8 = arith.constant 8 : index
          scf.for %i = %c0 to %c4 step %c1_0 {
            scf.for %j = %c0 to %c8 step %c1_0 {
              %n = affine.apply #map()[%i, %j, %tx]
              vector.transfer_write %cst, %buf[%n, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<64x2x8x8xf32, 2>
            }
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<64x2x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// The tile index reaches the index through an arith.muli. The shrink's index
// rewrite only zeroes tile indices the index's own op reads, so `tile * 2`
// stays in the index and counts over both tiles: i + 2 * tile reaches row 3,
// the whole dimension.

// CHECK-LABEL: func.func @arith_index
// CHECK: memref.alloc() : memref<4x4x8x8xf32, 2>
// CHECK-NOT: air.shrinkage
// CHECK: return
module {
  func.func @arith_index() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<4x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<4x4x8x8xf32, 2>
          air.execute_terminator %a : memref<4x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<4x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          scf.for %i = %c0 to %c2_0 step %c1_0 {
            scf.for %j = %c0 to %c2_0 step %c1_0 {
              %ty2 = arith.muli %ty, %c2_0 : index
              %tx2 = arith.muli %tx, %c2_0 : index
              %n = arith.addi %i, %ty2 : index
              %m = arith.addi %j, %tx2 : index
              vector.transfer_write %cst, %buf[%n, %m, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<4x4x8x8xf32, 2>
            }
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<4x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// The same arith ops inside an air.execute: the rewrite zeroes the tile indices
// its body reads, so the index becomes the IV and the buffer shrinks.

// CHECK-LABEL: func.func @execute_arith_index
// CHECK: memref.alloc() {air.shrinkage = true} : memref<2x2x8x8xf32, 2>
// CHECK: air.execute -> (index) {
// CHECK-NEXT: air.execute_terminator %{{.*}} : index
module {
  func.func @execute_arith_index() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<4x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<4x4x8x8xf32, 2>
          air.execute_terminator %a : memref<4x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<4x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          scf.for %i = %c0 to %c2_0 step %c1_0 {
            scf.for %j = %c0 to %c2_0 step %c1_0 {
              %tn, %n = air.execute -> (index) {
                %ty2 = arith.muli %ty, %c2_0 : index
                %v = arith.addi %i, %ty2 : index
                air.execute_terminator %v : index
              }
              %tm, %m = air.execute -> (index) {
                %tx2 = arith.muli %tx, %c2_0 : index
                %v = arith.addi %j, %tx2 : index
                air.execute_terminator %v : index
              }
              vector.transfer_write %cst, %buf[%n, %m, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<4x4x8x8xf32, 2>
            }
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<4x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// The outer loop's trip count is only known at run time (the herd size), so
// that dimension keeps its full size; the inner one still shrinks.

// CHECK-LABEL: func.func @dynamic_trip_count
// CHECK: memref.alloc() {air.shrinkage = true} : memref<4x2x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
module {
  func.func @dynamic_trip_count() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<4x4x8x8xf32, 2>) {
          %a = memref.alloc() : memref<4x4x8x8xf32, 2>
          air.execute_terminator %a : memref<4x4x8x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<4x4x8x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c2_0 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf32>
          scf.for %i = %c0 to %sy step %c1_0 {
            scf.for %j = %c0 to %c2_0 step %c1_0 {
              %n = affine.apply #map()[%i, %ty]
              %m = affine.apply #map()[%j, %tx]
              vector.transfer_write %cst, %buf[%n, %m, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<4x4x8x8xf32, 2>
            }
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<4x4x8x8xf32, 2>
        }
      }
    }
    return
  }
}
