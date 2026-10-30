//===- shim_read_back_pipeline.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file %s -air-renumber-dma -air-to-aie="device=npu2 row-offset=2 col-offset=0" -air-opt-shim-dma-bds="device=npu2" -canonicalize -cse -air-to-std -symbol-dce -affine-expand-index-ops -canonicalize -cse -airrt-to-npu | FileCheck %s
// RUN: air-opt -split-input-file %s -air-isolate-async-dma-loop-nests="scope=launch" -air-renumber-dma -air-to-aie="device=npu2 row-offset=2 col-offset=0" -air-opt-shim-dma-bds="device=npu2" -canonicalize -cse -air-to-std -symbol-dce -affine-expand-index-ops -canonicalize -cse -airrt-to-npu | FileCheck %s
// RUN: air-opt -split-input-file %s -air-renumber-dma -air-to-aie="device=npu2 row-offset=2 col-offset=0" -air-opt-shim-dma-bds="device=npu2" | FileCheck %s --check-prefix=IDS

// Four launch iterations each contain a producing job and a read-back job.
// The same static channel ops execute each iteration. Keep all three producer
// B feeds before the read-back wait, and consume the drain token in the loop
// body before starting the next iteration. Cover both isolation entry points.
// CHECK-LABEL: aie.runtime_sequence @jobs
// CHECK: scf.for
// CHECK: aiex.dma_configure_task_for @air_a
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: %[[D:.*]] = aiex.dma_configure_task_for @air_out
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_await_task(%[[D]])
// CHECK: aiex.dma_configure_task_for @air_a
// CHECK: aie.dma_bd(%arg2
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK-NOT: aiex.dma_await_task(%[[D]])
// CHECK: }

module {
  func.func private @compute(memref<64xi32, 2>)
  air.channel @a [1]
  air.channel @b [1]
  air.channel @out [1]
  func.func @jobs(%input: memref<256xi32>, %weights: memref<256xi32>, %output: memref<256xi32>) {
    %c4 = arith.constant 4 : index
    air.launch (%i) in (%n=%c4) args(%x=%input, %w=%weights, %y=%output) : memref<256xi32>, memref<256xi32>, memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %t0 = air.channel.put async @a[%c0] (%x[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t1 = air.channel.put async @b[%c0] (%w[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t2 = air.channel.put async @b[%c0] (%x[%c64] [%c64] [%c1]) : (memref<256xi32>)
      %t3 = air.channel.put async @b[%c0] (%w[%c128] [%c64] [%c1]) : (memref<256xi32>)
      %d = air.channel.get async @out[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %r = air.channel.put async [%d] @a[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t4 = air.channel.put async @b[%c0] (%w[%c64] [%c64] [%c1]) : (memref<256xi32>)
      air.segment @seg {
        %one = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%sx=%one, %sy=%one) {
          %zero = arith.constant 0 : index
          %l = memref.alloc() : memref<64xi32, 2>
          air.channel.get @a[%zero] (%l[] [] []) : (memref<64xi32, 2>)
          air.channel.get @b[%zero] (%l[] [] []) : (memref<64xi32, 2>)
          func.call @compute(%l) : (memref<64xi32, 2>) -> ()
          air.channel.put @out[%zero] (%l[] [] []) : (memref<64xi32, 2>)
        }
      }
    }
    return
  }
}

// -----

// IDS-LABEL: func.func @cloned_jobs
// IDS: air.channel.put async {{.*}}@a{{.*}}id = [[ID:[0-9]+]] : i32
// IDS: air.channel.get async {{.*}}@out
// IDS: air.channel.put async {{.*}}@a
// IDS: air.channel.put async {{.*}}@a{{.*}}id = [[ID]] : i32

// BD specialization clones two jobs after air-renumber-dma. Repeated static
// IDs must not change the order of either copy or move the second job's feeds
// before the first job's read-back barrier.
// CHECK-LABEL: aie.runtime_sequence @cloned_jobs
// CHECK: %[[CD0:.*]] = aiex.dma_configure_task_for @air_out
// CHECK: aiex.dma_configure_task_for @air_a
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_await_task(%[[CD0]])
// CHECK: aiex.dma_configure_task_for @air_a
// CHECK: %[[CD1:.*]] = aiex.dma_configure_task_for @air_out
// CHECK: aiex.dma_configure_task_for @air_a
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_configure_task_for @air_b
// CHECK: aiex.dma_await_task(%[[CD1]])
// CHECK: aiex.dma_configure_task_for @air_a

module {
  func.func private @compute(memref<64xi32, 2>)
  air.channel @a [1]
  air.channel @b [1]
  air.channel @out [1]
  func.func @cloned_jobs(%input: memref<256xi32>, %weights: memref<256xi32>, %output: memref<256xi32>) {
    %c4 = arith.constant 1 : index
    air.launch (%i) in (%n=%c4) args(%x=%input, %w=%weights, %y=%output) : memref<256xi32>, memref<256xi32>, memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c2 = arith.constant 2 : index
      scf.for %job = %c0 to %c2 step %c1 {
      %t0 = air.channel.put async @a[%c0] (%x[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t1 = air.channel.put async @b[%c0] (%w[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t2 = air.channel.put async @b[%c0] (%x[%c64] [%c64] [%c1]) : (memref<256xi32>)
      %t3 = air.channel.put async @b[%c0] (%w[%c128] [%c64] [%c1]) : (memref<256xi32>)
      %d = air.channel.get async @out[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %r = air.channel.put async [%d] @a[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t4 = air.channel.put async @b[%c0] (%w[%c64] [%c64] [%c1]) : (memref<256xi32>)
      }
      air.segment @seg {
        %one = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%sx=%one, %sy=%one) {
          %zero = arith.constant 0 : index
          %l = memref.alloc() : memref<64xi32, 2>
          air.channel.get @a[%zero] (%l[] [] []) : (memref<64xi32, 2>)
          air.channel.get @b[%zero] (%l[] [] []) : (memref<64xi32, 2>)
          func.call @compute(%l) : (memref<64xi32, 2>) -> ()
          air.channel.put @out[%zero] (%l[] [] []) : (memref<64xi32, 2>)
        }
      }
    }
    return
  }
}
