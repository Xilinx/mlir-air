//===- air_gpu_outlining.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining | FileCheck %s

// Verifies that air-gpu-outlining (after air-to-rocdl) outlines the GPU
// kernel body into a gpu.module + gpu.func:
//   gpu.launch { body } -> gpu.launch_func @module::@func
//   Kernel body moved to gpu.func with gpu.kernel attribute

// CHECK: gpu.launch_func @{{.*}}::@{{.*}} blocks in
// CHECK: gpu.module @
// CHECK: gpu.func @{{.*}} kernel
// CHECK: gpu.return

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
