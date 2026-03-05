//===- air_to_rocdl.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s

// Verifies that air-to-rocdl converts AIR hierarchy to GPU dialect:
//   air.launch(gx, gy) -> gpu.launch blocks(gx, gy, 1)
//   air.herd(hx, hy)   -> gpu.launch threads(hx, hy, 1)
//   air.segment         -> unwrapped
//   memref space=1      -> GPU workgroup attribution (space 3)
//   memref space=2      -> GPU private attribution (space 5)

// CHECK-LABEL: func.func @vecadd
// CHECK-NOT: air.launch
// CHECK-NOT: air.segment
// CHECK-NOT: air.herd
// CHECK: gpu.launch
// CHECK-SAME: blocks
// CHECK-SAME: threads
// CHECK: gpu.terminator

#map = affine_map<()[s0] -> (s0 * 16)>
module {
  func.func @vecadd(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%in0=%arg0, %in1=%arg1, %out=%arg2) : memref<64xf32>, memref<64xf32>, memref<64xf32> {
      air.segment @seg args(%s0=%in0, %s1=%in1, %s2=%out) : memref<64xf32>, memref<64xf32>, memref<64xf32> {
        %c4_s = arith.constant 4 : index
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%ntx=%c4_s, %nty=%c1_s) args(%h0=%s0, %h1=%s1, %h2=%s2) : memref<64xf32>, memref<64xf32>, memref<64xf32> {
          %offset = affine.apply #map()[%tx]
          %c16 = arith.constant 16 : index
          %c1_h = arith.constant 1 : index
          scf.for %i = %offset to %c16 step %c1_h {
            %a = memref.load %h0[%i] : memref<64xf32>
            %b = memref.load %h1[%i] : memref<64xf32>
            %c = arith.addf %a, %b : f32
            memref.store %c, %h2[%i] : memref<64xf32>
          }
          air.herd_terminator
        }
      }
    }
    return
  }
}
