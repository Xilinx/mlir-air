//===- free_awaited_runtime_task.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// A task configured in a runtime loop gets its BD ids from a runtime pool that
// only dma_free_task returns them to, so it is freed right after its await.
// CHECK-LABEL: aie.runtime_sequence @in_loop
// CHECK: scf.for
// CHECK: %[[T:.*]] = aiex.dma_configure_task_for @out0
// CHECK: aiex.dma_await_task(%[[T]])
// CHECK-NEXT: aiex.dma_free_task(%[[T]])
// CHECK-NEXT: }
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @out0(%t0, S2MM, 0)
    aie.runtime_sequence @in_loop(%arg0: memref<256xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        %t = aiex.dma_configure_task_for @out0 {
          aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
      }
    }
  }
}

// -----

// The task is awaited in either arm of an scf.if and freed once after it. No
// other free is added, which would return the same BD ids twice.
// CHECK-LABEL: aie.runtime_sequence @free_after_if
// CHECK: %[[T:.*]] = aiex.dma_configure_task_for @out0
// CHECK: scf.if
// CHECK-NEXT: aiex.dma_await_task(%[[T]])
// CHECK-NEXT: } else {
// CHECK-NEXT: aiex.dma_await_task(%[[T]])
// CHECK-NEXT: }
// CHECK-NEXT: aiex.dma_free_task(%[[T]])
// CHECK-NOT: aiex.dma_free_task
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @out0(%t0, S2MM, 0)
    aie.runtime_sequence @free_after_if(%arg0: memref<256xi32>, %n: index, %c: i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        %t = aiex.dma_configure_task_for @out0 {
          aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%t)
        scf.if %c {
          aiex.dma_await_task(%t)
        } else {
          aiex.dma_await_task(%t)
        }
        aiex.dma_free_task(%t)
      }
    }
  }
}

// -----

// A task configured at the top level of the sequence gets its BD ids at
// compile time, and the await releases them: no free is added.
// CHECK-LABEL: aie.runtime_sequence @top_level
// CHECK: %[[T:.*]] = aiex.dma_configure_task_for @out0
// CHECK: aiex.dma_await_task(%[[T]])
// CHECK-NOT: aiex.dma_free_task
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @out0(%t0, S2MM, 0)
    aie.runtime_sequence @top_level(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
