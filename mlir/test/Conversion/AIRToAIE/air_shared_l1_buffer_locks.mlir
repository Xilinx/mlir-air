//===- air_shared_l1_buffer_locks.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Tests for shared L1 buffer lock allocation in AIRToAIE conversion.
// Shared L1 buffers enable inter-core communication via shared L1 memory
// between neighboring AIE tiles.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' --split-input-file | FileCheck %s

// Test basic shared L1 buffer between two herds: one producer, one consumer.
// The shared buffer is allocated at segment level and passed to both herds.
// Expected:
// - Single aie.buffer for shared L1 memory
// - Producer lock with init=1 (producer can write first)
// - Consumer lock with init=0 (consumer waits for producer)
// - Producer core: acquire(prod_lock) -> write -> release(cons_lock)
// - Consumer core: acquire(cons_lock) -> read -> release(prod_lock)

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[TILE0:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[TILE1:.*]] = aie.tile(0, 3)
// CHECK-DAG: %[[CONS_LOCK:.*]] = aie.lock(%[[TILE1]], {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD_LOCK:.*]] = aie.lock(%[[TILE1]], {{.*}}) {init = 1 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CHECK-DAG: %[[SHARED_BUF:.*]] = aie.buffer(%[[TILE1]]) {sym_name = "shared_l1{{.*}}"} : memref<64x64xbf16, 2>

// Check consumer core (tile_0_3) appears first in output
// CHECK: aie.core(%[[TILE1]])
// CHECK: aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%[[PROD_LOCK]], Release, 1)

// Check producer core (tile_0_2) appears second
// CHECK: aie.core(%[[TILE0]])
// CHECK: aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK: vector.transfer_write {{.*}}, %[[SHARED_BUF]]
// CHECK: aie.use_lock(%[[CONS_LOCK]], Release, 1)

module {
  func.func @shared_l1_producer_consumer() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) {
      air.segment @segment_0 {
        // Shared L1 buffer allocated at segment level (outside both herds)
        %alloc_shared = memref.alloc() : memref<64x64xbf16, 2>
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index

        // Producer herd: writes to shared buffer
        air.herd @herd_producer tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<64x64xbf16, 2> attributes {x_loc=0, y_loc=2} {
          %cst = arith.constant 1.0 : bf16
          %c0_2 = arith.constant 0 : index
          %v = vector.broadcast %cst : bf16 to vector<16xbf16>
          vector.transfer_write %v, %shared_buf[%c0_2, %c0_2] {in_bounds = [true]} : vector<16xbf16>, memref<64x64xbf16, 2>
          air.herd_terminator
        }

        // Consumer herd: reads from shared buffer
        air.herd @herd_consumer tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<64x64xbf16, 2> attributes {x_loc=0, y_loc=3} {
          %cst = arith.constant 0.0 : bf16
          %c0_3 = arith.constant 0 : index
          %v = vector.transfer_read %shared_buf[%c0_3, %c0_3], %cst {in_bounds = [true]} : memref<64x64xbf16, 2>, vector<16xbf16>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Test shared L1 buffer with WRITE-ONLY access pattern (two herds both write).
// This triggers the deadlock avoidance logic: since there are no consumers,
// using producer/consumer locks would deadlock. Instead, a single mutex lock
// is allocated for mutual exclusion.
// Expected warning: "shared L1 buffer has write-only access pattern"

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[TILE0:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[TILE1:.*]] = aie.tile(0, 3)
// CHECK-DAG: %[[MUTEX_LOCK:.*]] = aie.lock({{.*}}) {init = 1 : i32, sym_name = "shared_l1{{.*}}_mutex_lock"}
// CHECK-DAG: %[[SHARED_BUF:.*]] = aie.buffer({{.*}}) {sym_name = "shared_l1{{.*}}"} : memref<32x32xi32, 2>

// Verify mutex strategy: acquire and release SAME lock (not cross-release)
// CHECK: aie.core
// CHECK: aie.use_lock(%[[MUTEX_LOCK]], AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%[[MUTEX_LOCK]], Release, 1)

module {
  func.func @shared_l1_single_herd_producer_only() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) {
      air.segment @segment_0 {
        // Shared L1 buffer allocated at segment level
        %alloc_shared = memref.alloc() : memref<32x32xi32, 2>
        %c1_1 = arith.constant 1 : index

        // Only herd_a writes to the buffer
        air.herd @herd_a tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<32x32xi32, 2> attributes {x_loc=0, y_loc=2} {
          %c0_2 = arith.constant 0 : index
          %cst = arith.constant 42 : i32
          %v = vector.broadcast %cst : i32 to vector<8xi32>
          vector.transfer_write %v, %shared_buf[%c0_2, %c0_2] {in_bounds = [true]} : vector<8xi32>, memref<32x32xi32, 2>
          air.herd_terminator
        }

        // herd_b also writes - should get producer locks too
        air.herd @herd_b tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<32x32xi32, 2> attributes {x_loc=0, y_loc=3} {
          %c0_3 = arith.constant 0 : index
          %cst = arith.constant 99 : i32
          %v = vector.broadcast %cst : i32 to vector<8xi32>
          vector.transfer_write %v, %shared_buf[%c0_3, %c0_3] {in_bounds = [true]} : vector<8xi32>, memref<32x32xi32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Test shared L1 buffer accessed via subview (view aliasing).
// The lock allocation should track view aliases and protect all accesses
// through the original buffer and its views.

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[TILE0:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[TILE1:.*]] = aie.tile(0, 3)
// CHECK-DAG: %[[CONS_LOCK:.*]] = aie.lock({{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD_LOCK:.*]] = aie.lock({{.*}}) {init = 1 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CHECK-DAG: %[[SHARED_BUF:.*]] = aie.buffer({{.*}}) {sym_name = "shared_l1{{.*}}"} : memref<64x64xf32, 2>

// Check producer core protects write through subview
// CHECK: aie.core
// CHECK: aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK: memref.subview %[[SHARED_BUF]]
// CHECK: aie.use_lock({{.*}}, Release, 1)

module {
  func.func @shared_l1_with_subview() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) {
      air.segment @segment_0 {
        %alloc_shared = memref.alloc() : memref<64x64xf32, 2>
        %c1_1 = arith.constant 1 : index

        // Producer herd: writes to shared buffer via subview
        air.herd @herd_producer tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<64x64xf32, 2> attributes {x_loc=0, y_loc=2} {
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          // Create a subview of the shared buffer
          %subview = memref.subview %shared_buf[0, 0][16, 16][1, 1] : memref<64x64xf32, 2> to memref<16x16xf32, strided<[64, 1]>, 2>
          %cst = arith.constant 3.14 : f32
          %v = vector.broadcast %cst : f32 to vector<16xf32>
          vector.transfer_write %v, %subview[%c0_2, %c0_2] {in_bounds = [true]} : vector<16xf32>, memref<16x16xf32, strided<[64, 1]>, 2>
          air.herd_terminator
        }

        // Consumer herd: reads from shared buffer directly
        air.herd @herd_consumer tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<64x64xf32, 2> attributes {x_loc=0, y_loc=3} {
          %cst = arith.constant 0.0 : f32
          %c0_3 = arith.constant 0 : index
          %v = vector.transfer_read %shared_buf[%c0_3, %c0_3], %cst {in_bounds = [true]} : memref<64x64xf32, 2>, vector<16xf32>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Test that LOCAL L1 buffers (allocated inside a single herd) do NOT get
// producer/consumer locks. Only SHARED L1 buffers (allocated at segment level
// and used by multiple herds) should get these locks.

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[TILE:.*]] = aie.tile(0, 2)
// CHECK: %[[LOCAL_BUF:.*]] = aie.buffer(%[[TILE]]) {sym_name = "buf{{.*}}"} : memref<16x16xi32, 2>

// Local buffers should NOT have prod/cons locks with "shared_l1" prefix
// CHECK-NOT: shared_l1{{.*}}_prod_lock
// CHECK-NOT: shared_l1{{.*}}_cons_lock

module {
  func.func @local_l1_no_shared_locks() {
    %c1 = arith.constant 1 : index
    air.herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) attributes {x_loc=0, y_loc=2} {
      // This is a LOCAL L1 buffer (allocated inside the herd)
      %local_buf = memref.alloc() : memref<16x16xi32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 123 : i32
      %v = vector.broadcast %cst : i32 to vector<8xi32>
      vector.transfer_write %v, %local_buf[%c0, %c0] {in_bounds = [true]} : vector<8xi32>, memref<16x16xi32, 2>
      air.herd_terminator
    }
    return
  }
}
