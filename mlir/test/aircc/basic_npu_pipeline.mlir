//===- basic_npu_pipeline.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that the C++ aircc binary correctly runs the full NPU compilation
// pipeline (placement, AIR-to-AIE, NPU lowering) and generates correct
// intermediate IR files.
//
// Note: aiecc backend may fail without peano, but the aircc pass pipeline
// still generates the intermediate files we want to verify.

// RUN: rm -rf %t && mkdir -p %t
// RUN: aircc %s --device=npu1 --tmpdir=%t --output-format=none -v 2>&1 || true

// The placed IR should contain air.channel declarations (from air-dma-to-channel)
// RUN: FileCheck %s --input-file=%t/placed.basic_npu_pipeline.mlir --check-prefix=PLACED

// The AIE IR should contain aie.device and aie.tile (from air-to-aie)
// RUN: FileCheck %s --input-file=%t/aie.basic_npu_pipeline.mlir --check-prefix=AIE

// The NPU IR should have no air.launch/air.segment ops (from air-to-std)
// RUN: FileCheck %s --input-file=%t/npu.basic_npu_pipeline.mlir --check-prefix=NPU

// PLACED: air.channel
// PLACED: air.channel.put
// PLACED: air.channel.get

// AIE: aie.device(npu1)
// AIE: aie.tile
// AIE: aie.lock
// AIE: aie.buffer

// NPU: aie.device(npu1)
// NPU-NOT: air.launch
// NPU-NOT: air.segment
// NPU-NOT: air.herd

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
