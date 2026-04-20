//===- non_npu_device.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that aircc handles non-NPU device (xcvc1902 Versal) targets.
// The optimization passes (dependency, dma-to-channel, etc.) should be
// skipped for non-NPU targets. The placed IR should still be generated.

// RUN: rm -rf %t && mkdir -p %t
// RUN: aircc %s --device=xcvc1902 --tmpdir=%t --output-format=none 2>&1 || true

// The placed IR should exist and contain air.herd with placement
// RUN: FileCheck %s --input-file=%t/placed.non_npu_device.mlir --check-prefix=PLACED

// The AIE IR should target xcvc1902
// RUN: FileCheck %s --input-file=%t/aie.non_npu_device.mlir --check-prefix=AIE

// PLACED: air.herd
// AIE: aie.device(xcvc1902)

module {
  func.func @copy(%arg0: memref<1024xui8>, %arg1: memref<1024xui8>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<1024xui8>, memref<1024xui8> {
      air.segment @seg  args(%arg4=%arg2, %arg5=%arg3) : memref<1024xui8>, memref<1024xui8> {
        %c1 = arith.constant 1 : index
        air.herd @herd  tile (%arg6, %arg7) in (%arg8=%c1, %arg9=%c1) args(%arg10=%arg4, %arg11=%arg5) : memref<1024xui8>, memref<1024xui8> {
          %c0 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_0 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<1024xui8, 2 : i32>
          air.dma_memcpy_nd (%alloc[] [] [], %arg10[%c0] [%c1024] [%c1_0]) : (memref<1024xui8, 2 : i32>, memref<1024xui8>)
          air.dma_memcpy_nd (%arg11[%c0] [%c1024] [%c1_0], %alloc[] [] []) : (memref<1024xui8>, memref<1024xui8, 2 : i32>)
          memref.dealloc %alloc : memref<1024xui8, 2 : i32>
        }
      }
    }
    return
  }
}
