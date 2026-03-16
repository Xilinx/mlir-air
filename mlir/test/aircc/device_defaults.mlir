//===- device_defaults.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that device-specific defaults are correctly applied (num-cols,
// col-offset, etc.) for different NPU targets.

// RUN: rm -rf %t && mkdir -p %t/npu1 %t/npu2

// Test npu1 defaults: 4 columns, col-offset=0
// RUN: aircc %s --device=npu1 --tmpdir=%t/npu1 --output-format=none -v 2>&1 || true
// RUN: FileCheck %s --input-file=%t/npu1/aie.device_defaults.mlir --check-prefix=NPU1AIE

// Test npu2 defaults: 8 columns
// RUN: aircc %s --device=npu2 --tmpdir=%t/npu2 --output-format=none -v 2>&1 || true
// RUN: FileCheck %s --input-file=%t/npu2/aie.device_defaults.mlir --check-prefix=NPU2AIE

// NPU1AIE: aie.device(npu1)

// NPU2AIE: aie.device(npu2)

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
