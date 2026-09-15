//===- air_gpu_outlining_globals.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining | FileCheck %s

// gpu.module is its own symbol table, so a memref.get_global that resolved at
// module scope dangles once outlining moves the body inside it -- the verifier
// reports "does not reference a valid global memref". Outlining must carry the
// definition across with the body.
//
// This matters because a module-scope global is how kernel-side device scratch
// is expressed: the per-chiplet rank counters need exactly this.

// The definition stays at module scope for any host-side user...
// CHECK:       memref.global "private" @scratch
// ...and a copy lands inside the gpu.module next to the kernel that uses it.
// CHECK:       gpu.module @k_module
// CHECK-DAG:     gpu.func @k_module
// CHECK-DAG:     memref.get_global @scratch
// CHECK-DAG:     memref.global "private" @scratch

module {
  memref.global "private" @scratch : memref<64xi32> = uninitialized
  func.func @k(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%out=%arg0) : memref<8xf32> {
      air.segment @seg args(%s0=%out) : memref<8xf32> {
        %c1_s = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %g = memref.get_global @scratch : memref<64xi32>
        %v = memref.load %g[%c0] : memref<64xi32>
        %f = arith.sitofp %v : i32 to f32
        memref.store %f, %s0[%c0] : memref<8xf32>
        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
