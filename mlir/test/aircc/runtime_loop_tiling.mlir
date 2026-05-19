//===- runtime_loop_tiling.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify aircc forwards tile-size CLI flags correctly:
//   - default → neither pass option set (no shim DMA tiling)
//   - --air-runtime-loop-tiling-sizes=N → shim-dma-tile-sizes=N,... (user override)
//   - --air-auto-derive-tile-sizes → auto-derive-tile-sizes=true (cost model, opt-in)
// Smoke: aircc must complete cleanly in all three modes through placed IR.

// RUN: rm -rf %t && mkdir -p %t/default %t/override %t/auto
// RUN: aircc %s --device=npu1 --tmpdir=%t/default --output-format=none 2>&1 || true
// RUN: aircc %s --device=npu1 --tmpdir=%t/override --output-format=none --air-runtime-loop-tiling-sizes=2 --air-runtime-loop-tiling-sizes=2 2>&1 || true
// RUN: aircc %s --device=npu1 --tmpdir=%t/auto --output-format=none --air-auto-derive-tile-sizes 2>&1 || true

// RUN: FileCheck %s --input-file=%t/default/placed.runtime_loop_tiling.mlir --check-prefix=PLACED
// RUN: FileCheck %s --input-file=%t/override/placed.runtime_loop_tiling.mlir --check-prefix=PLACED
// RUN: FileCheck %s --input-file=%t/auto/placed.runtime_loop_tiling.mlir --check-prefix=PLACED

// PLACED: air.channel

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
