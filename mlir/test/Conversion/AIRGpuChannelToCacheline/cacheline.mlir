//===- cacheline.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s --split-input-file -air-gpu-channel-to-cacheline | FileCheck %s
// RUN: air-opt %s --split-input-file -air-gpu-channel-to-cacheline -verify-diagnostics 2>&1 | FileCheck %s --check-prefix=ANY

// 1-put / 1-get gpu_symmetric_heap channel, both inside air.herd bodies
// nested in an air.rank scope (required by the op verifier for
// gpu_symmetric_heap channels). Producer rank 0, consumer rank 1.
// The pass should:
//   * Erase the air.channel symbol.
//   * Replace the put with air.translate + cooperative cacheline store.
//   * Replace the get with scf.while spin + gpu.shuffle idx.

// CHECK-LABEL: @cacheline_pair
// CHECK-NOT: air.channel @C
// Put expansion: per-lane via gpu.lane_id (AIR model: PE = wavefront, so
// the herd body runs once per PE and lanes are addressed via gpu.lane_id).
// CHECK: air.translate
// CHECK: gpu.lane_id
// CHECK: scf.if
// CHECK: memref.load
// CHECK: arith.select
// CHECK: memref.store
// Get expansion: zero-result scf.while + memref.atomic_rmw addi (upstream
// idiom; see mlir/test/Integration/GPU/CUDA/concurrent-kernels.mlir).
// The atomic_rmw's Write effect keeps the spin alive past DCE in
// subsequent passes, and encodes "observable read" semantics in the IR.
// CHECK: gpu.lane_id
// CHECK: scf.while
// CHECK: scf.if
// CHECK: memref.atomic_rmw addi
// CHECK: gpu.shuffle idx
// CHECK: arith.cmpi ne
// CHECK: scf.condition
air.channel @C [1] {channel_type = "gpu_symmetric_heap"}
func.func @cacheline_pair(%data_outer: memref<32xi32, #air.symmetric_heap>,
                          %bases_outer: memref<?xindex, #air.symmetric_heap>) {
  %c2 = arith.constant 2 : index
  air.rank (%rid) in (%rsize = %c2) args(%data = %data_outer, %bases = %bases_outer)
      : memref<32xi32, #air.symmetric_heap>, memref<?xindex, #air.symmetric_heap> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %is_p = arith.cmpi eq, %rid, %c0 : index
    scf.if %is_p {
      air.launch (%bx) in (%nx = %c1)
          args(%ld = %data, %lb = %bases)
          : memref<32xi32, #air.symmetric_heap>,
            memref<?xindex, #air.symmetric_heap> {
        air.segment args(%sd = %ld, %sb = %lb)
            : memref<32xi32, #air.symmetric_heap>,
              memref<?xindex, #air.symmetric_heap> {
          %c64 = arith.constant 64 : index
          %c1_s = arith.constant 1 : index
          air.herd tile (%tx, %ty) in (%ntx = %c64, %nty = %c1_s)
              args(%hd = %sd, %hb = %sb)
              : memref<32xi32, #air.symmetric_heap>,
                memref<?xindex, #air.symmetric_heap> {
            %c0_idx = arith.constant 0 : index
            air.channel.put @C[%c0_idx] (%hd[][][])
                : (memref<32xi32, #air.symmetric_heap>)
            air.herd_terminator
          }
          air.segment_terminator
        }
        air.launch_terminator
      }
    }
    %is_c = arith.cmpi eq, %rid, %c1 : index
    scf.if %is_c {
      air.launch (%bx) in (%nx = %c1)
          args(%ld = %data, %lb = %bases)
          : memref<32xi32, #air.symmetric_heap>,
            memref<?xindex, #air.symmetric_heap> {
        air.segment args(%sd = %ld, %sb = %lb)
            : memref<32xi32, #air.symmetric_heap>,
              memref<?xindex, #air.symmetric_heap> {
          %c64 = arith.constant 64 : index
          %c1_s = arith.constant 1 : index
          air.herd tile (%tx, %ty) in (%ntx = %c64, %nty = %c1_s)
              args(%hd = %sd, %hb = %sb)
              : memref<32xi32, #air.symmetric_heap>,
                memref<?xindex, #air.symmetric_heap> {
            %c0_idx = arith.constant 0 : index
            air.channel.get @C[%c0_idx] (%hd[][][])
                : (memref<32xi32, #air.symmetric_heap>)
            air.herd_terminator
          }
          air.segment_terminator
        }
        air.launch_terminator
      }
    }
    air.rank_terminator
  }
  return
}

// -----

// Non-gpu_symmetric_heap channels are left alone.
// CHECK: air.channel @Other
// CHECK-LABEL: @other_channel_type_noop
// CHECK: air.channel.put @Other
air.channel @Other [] {channel_type = "npu_dma_stream"}
func.func @other_channel_type_noop(%src: memref<32xi32>) {
  air.channel.put @Other[] (%src[][][]) : (memref<32xi32>)
  return
}
