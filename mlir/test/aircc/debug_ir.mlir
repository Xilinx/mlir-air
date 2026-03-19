//===- debug_ir.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that --debug-ir produces per-pass IR files and a pass.log with
// checkpoint markers, matching the Python aircc.py behavior.

// RUN: rm -rf %t && mkdir -p %t
// RUN: aircc %s --device=npu1 --tmpdir=%t --output-format=none --debug-ir -v 2>&1 | tee %t/stdout.log || true
// RUN: FileCheck %s --input-file=%t/stdout.log

// Verify per-pass IR files are created
// RUN: ls %t/debug_ir/pass_000_initial_input.mlir
// RUN: ls %t/debug_ir/pass_001_after_air-insert-launch-around-herd.mlir
// RUN: ls %t/debug_ir/pass_004_after_air-dependency.mlir
// RUN: ls %t/debug_ir/pass_008_after_air-dma-to-channel.mlir

// Verify pass.log exists and has correct structure
// RUN: FileCheck %s --input-file=%t/debug_ir/pass.log --check-prefix=LOG

// Verbose output should show per-pass execution
// CHECK: [DEBUG] Splitting pipeline into
// CHECK: [PASS 000] Saved initial IR
// CHECK: [PASS 001] air-insert-launch-around-herd
// CHECK: [PASS 004] air-dependency
// CHECK: [PASS 008] air-dma-to-channel

// Pass log should have checkpoints
// LOG: MLIR-AIR Compilation Pass Log
// LOG: [PASS 000] [Initial IR before passes]
// LOG: CHECKPOINT: AIR Placement Complete
// LOG: This is equivalent to: placed.air.mlir
// LOG: CHECKPOINT: AIR to AIE Conversion Complete
// LOG: This is equivalent to: aie.air.mlir
// LOG: CHECKPOINT: NPU Instruction Generation Complete
// LOG: This is equivalent to: npu.air.mlir

module {
  func.func @copy(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4096xui8>, memref<4096xui8> {
      air.segment @seg  args(%arg4=%arg2, %arg5=%arg3) : memref<4096xui8>, memref<4096xui8> {
        %c1 = arith.constant 1 : index
        air.herd @copyherd  tile (%arg6, %arg7) in (%arg8=%c1, %arg9=%c1) args(%arg10=%arg4, %arg11=%arg5) : memref<4096xui8>, memref<4096xui8> {
          %c0 = arith.constant 0 : index
          %c4096 = arith.constant 4096 : index
          %c1024 = arith.constant 1024 : index
          scf.for %arg12 = %c0 to %c4096 step %c1024 {
            %alloc = memref.alloc() : memref<1024xui8, 2 : i32>
            %alloc_1 = memref.alloc() : memref<1024xui8, 2 : i32>
            %c1_2 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc[] [] [], %arg10[%arg12] [%c1024] [%c1_2]) : (memref<1024xui8, 2 : i32>, memref<4096xui8>)
            %c1_3 = arith.constant 1 : index
            scf.for %arg13 = %c0 to %c1024 step %c1_3 {
              %0 = memref.load %alloc[%arg13] : memref<1024xui8, 2 : i32>
              memref.store %0, %alloc_1[%arg13] : memref<1024xui8, 2 : i32>
            }
            air.dma_memcpy_nd (%arg11[%arg12] [%c1024] [%c1_2], %alloc_1[] [] []) : (memref<4096xui8>, memref<1024xui8, 2 : i32>)
            memref.dealloc %alloc : memref<1024xui8, 2 : i32>
            memref.dealloc %alloc_1 : memref<1024xui8, 2 : i32>
          }
        }
      }
    }
    return
  }
}
