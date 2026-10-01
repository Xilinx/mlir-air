//===- shrink_memref_sizes_by_access_affine_apply.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file | FileCheck %s

// A segment-level L1 buffer reached only by vector transfers inside the herd,
// at an index that is a bare affine.apply of a loop IV and the herd tile index
// -- the shape of a GEMM accumulator whose epilogue runs on the cores rather
// than as a DMA. Each core walks a 2x2 block of 8x8 tiles, so the per-core
// buffer is 2x2x8x8. The loop's range has to count: sized from one transfer's
// vector alone it would be 1x1x8x8, and every tile would alias onto it.

// CHECK-LABEL: func.func @iv_plus_tile
// CHECK: air.execute -> (memref<2x2x8x8xf32, 2>)
// CHECK: memref.alloc() {air.shrinkage = true} : memref<2x2x8x8xf32, 2>
// CHECK: air.herd
// CHECK: %[[N:.*]] = affine.apply #{{.*}}()[%{{.*}}, %c0]
// CHECK: %[[M:.*]] = affine.apply #{{.*}}()[%{{.*}}, %c0]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[N]], %[[M]], %c0{{.*}}, %c0{{.*}}] {{.*}} : vector<1x1x8x8xf32>, memref<2x2x8x8xf32, 2>
#map = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
module {
  func.func @iv_plus_tile() {
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

// An index that is not linear in the IV has no single stride: `s0 + s0
// floordiv 4` steps by 1 between s0 = 0 and 1, but over 8 iterations reaches
// 9, so an extent of 8 would be too small. The dimension is left whole.

// CHECK-LABEL: func.func @iv_floordiv
// CHECK: memref.alloc() : memref<20x8xf32, 2>
// CHECK-NOT: air.shrinkage
// CHECK: return
#map1 = affine_map<()[s0, s1] -> (s0 + s0 floordiv 4 + s1 * 10)>
module {
  func.func @iv_floordiv() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 {
        %c2 = arith.constant 2 : index
        %t0, %alloc = air.execute -> (memref<20x8xf32, 2>) {
          %a = memref.alloc() : memref<20x8xf32, 2>
          air.execute_terminator %a : memref<20x8xf32, 2>
        }
        %t1 = air.herd @herd_0 async [%t0] tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%buf=%alloc) : memref<20x8xf32, 2> {
          %c0 = arith.constant 0 : index
          %c1_0 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
          scf.for %i = %c0 to %c8 step %c1_0 {
            %n = affine.apply #map1()[%i, %tx]
            vector.transfer_write %cst, %buf[%n, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<20x8xf32, 2>
          }
        }
        %t2 = air.execute [%t1] {
          memref.dealloc %alloc : memref<20x8xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// Two loop IVs folded into one index span the product of their ranges, which
// one size and stride cannot describe. The dimension is left whole.

// CHECK-LABEL: func.func @two_ivs
// CHECK: memref.alloc() : memref<64x8xf32, 2>
// CHECK-NOT: air.shrinkage
// CHECK: return
#map2 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 4 + s2 * 32)>
module {
  func.func @two_ivs() {
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
          %c1_0 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c8 = arith.constant 8 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x8xf32>
          scf.for %i = %c0 to %c4 step %c1_0 {
            scf.for %j = %c0 to %c8 step %c1_0 {
              %n = affine.apply #map2()[%i, %j, %tx]
              vector.transfer_write %cst, %buf[%n, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<64x8xf32, 2>
            }
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
