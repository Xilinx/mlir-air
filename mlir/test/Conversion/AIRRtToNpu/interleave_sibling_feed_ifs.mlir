//===- interleave_sibling_feed_ifs.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// Two guarded bursts of feeds, each on its own channel, the second issued
// behind the first's awaits: one if, the tasks alternating. The feed between
// them goes ahead.
// CHECK-LABEL: aie.runtime_sequence @two_columns
// CHECK: aiex.dma_configure_task_for @mid
// CHECK: scf.if
// CHECK: aiex.dma_configure_task_for @col0
// CHECK: aiex.dma_configure_task_for @col1
// CHECK: aiex.dma_configure_task_for @col0
// CHECK: aiex.dma_configure_task_for @col1
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_configure_task_for @col0
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_configure_task_for @col1
// CHECK-NOT: scf.if
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @col0(%t0, MM2S, 0)
    aie.shim_dma_allocation @col1(%t1, MM2S, 0)
    aie.shim_dma_allocation @mid(%t1, MM2S, 1)
    aie.runtime_sequence @two_columns(%a: memref<256xi32>, %b: memref<256xi32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %p0 = arith.cmpi sgt, %n, %c0 : i32
      scf.if %p0 {
        %x0 = aiex.dma_configure_task_for @col0 {
          aie.dma_bd(%a : memref<256xi32> offset = 0 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%x0)
        %x1 = aiex.dma_configure_task_for @col0 {
          aie.dma_bd(%a : memref<256xi32> offset = 64 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%x1)
        aiex.dma_await_task(%x0)
        %x2 = aiex.dma_configure_task_for @col0 {
          aie.dma_bd(%a : memref<256xi32> offset = 128 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%x2)
      }
      %m = aiex.dma_configure_task_for @mid {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%m)
      %p1 = arith.cmpi sgt, %n, %c0 : i32
      scf.if %p1 {
        %y0 = aiex.dma_configure_task_for @col1 {
          aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%y0)
        %y1 = aiex.dma_configure_task_for @col1 {
          aie.dma_bd(%b : memref<256xi32> offset = 64 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%y1)
        aiex.dma_await_task(%y0)
        %y2 = aiex.dma_configure_task_for @col1 {
          aie.dma_bd(%b : memref<256xi32> offset = 128 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%y2)
      }
    }
  }
}

// -----

// Both bursts on one channel: left in order.
// CHECK-LABEL: aie.runtime_sequence @one_channel
// CHECK: scf.if
// CHECK: scf.if
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @col0(%t0, MM2S, 0)
    aie.runtime_sequence @one_channel(%a: memref<256xi32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %p = arith.cmpi sgt, %n, %c0 : i32
      scf.if %p {
        %x0 = aiex.dma_configure_task_for @col0 {
          aie.dma_bd(%a : memref<256xi32> offset = 0 len = 64)
          aie.end
        }
        aiex.dma_start_task(%x0)
      }
      scf.if %p {
        %y0 = aiex.dma_configure_task_for @col0 {
          aie.dma_bd(%a : memref<256xi32> offset = 64 len = 64)
          aie.end
        }
        aiex.dma_start_task(%y0)
      }
    }
  }
}

// -----

// A drain in one: left in order.
// CHECK-LABEL: aie.runtime_sequence @with_drain
// CHECK: scf.if
// CHECK: scf.if
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @col0(%t0, MM2S, 0)
    aie.shim_dma_allocation @out1(%t1, S2MM, 0)
    aie.runtime_sequence @with_drain(%a: memref<256xi32>, %b: memref<256xi32>, %n: i32) {
      %c0 = arith.constant 0 : i32
      %p = arith.cmpi sgt, %n, %c0 : i32
      scf.if %p {
        %x0 = aiex.dma_configure_task_for @col0 {
          aie.dma_bd(%a : memref<256xi32> offset = 0 len = 64)
          aie.end
        }
        aiex.dma_start_task(%x0)
      }
      scf.if %p {
        %y0 = aiex.dma_configure_task_for @out1 {
          aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%y0)
        aiex.dma_await_task(%y0)
      }
    }
  }
}
