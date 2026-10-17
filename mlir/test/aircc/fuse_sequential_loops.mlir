//===- fuse_sequential_loops.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-fuse-channels with sequential-loops runs after air-split-l2-memref only
// with --fuse-sequential-loops.

// RUN: rm -rf %t && mkdir -p %t/default %t/seq
// RUN: aircc %s --device=npu2 --tmpdir=%t/default --output-format=none --debug-ir 2>&1 || true
// RUN: aircc %s --device=npu2 --tmpdir=%t/seq --output-format=none --debug-ir --fuse-sequential-loops 2>&1 || true
// RUN: ls %t/default/debug_ir | FileCheck %s --check-prefix=DEFAULT
// RUN: ls %t/seq/debug_ir | FileCheck %s --check-prefix=SEQ

// DEFAULT: after_func.func_air-split-l2-memref
// DEFAULT-NOT: after_air-fuse-channels
// SEQ: after_func.func_air-split-l2-memref
// SEQ: after_air-fuse-channels

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
            %c1_2 = arith.constant 1 : index
            air.dma_memcpy_nd (%alloc[] [] [], %arg10[%arg12] [%c1024] [%c1_2]) : (memref<1024xui8, 2 : i32>, memref<4096xui8>)
            air.dma_memcpy_nd (%arg11[%arg12] [%c1024] [%c1_2], %alloc[] [] []) : (memref<4096xui8>, memref<1024xui8, 2 : i32>)
            memref.dealloc %alloc : memref<1024xui8, 2 : i32>
          }
        }
      }
    }
    return
  }
}
