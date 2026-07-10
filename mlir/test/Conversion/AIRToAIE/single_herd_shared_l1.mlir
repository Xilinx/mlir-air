//===- single_herd_shared_l1.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Tests intra-herd (core-to-core) shared L1 buffer lowering in AIRToAIE.
// A SINGLE air.herd contains multiple cores that hand data to each other
// through a segment-scope L1 buffer passed as a herd operand. Per-core role
// (writer vs reader) is selected from the tile index via scf.index_switch.
// Unlike the multi-herd producer/consumer case, the shared buffer is inferred
// from a cross-core read-after-write dependence (some reader core is not a
// writer core), not from the buffer being used by more than one herd.

// RUN: air-opt %s -air-to-aie='device=npu1 row-offset=2' --split-input-file | FileCheck %s

// A [1,2] herd: core ty=0 writes the shared buffer, core ty=1 reads it.
// Expected:
// - A single shared aie.buffer (on the reader tile), NOT a per-core local copy.
// - Producer lock init=1, consumer lock init=0.
// - Writer core: acquire(prod_lock) -> write -> release(cons_lock).
// - Reader core: acquire(cons_lock) -> release(prod_lock).
// - No aie.flow between the two compute tiles (data moves via shared L1, not DMA).

// CHECK-LABEL: aie.device
// CHECK-DAG: %[[WRITER:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[READER:.*]] = aie.tile(0, 3)
// CHECK-DAG: %[[CONS_LOCK:.*]] = aie.lock(%[[READER]], {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD_LOCK:.*]] = aie.lock(%[[READER]], {{.*}}) {init = 1 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CHECK-DAG: %[[SHARED_BUF:.*]] = aie.buffer(%[[READER]]) {sym_name = "shared_l1{{.*}}"} : memref<64xbf16, 2>

// CHECK-NOT: aie.flow

// Reader core (tile_0_3) is emitted first.
// CHECK: aie.core(%[[READER]])
// CHECK: aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%[[PROD_LOCK]], Release, 1)

// Writer core (tile_0_2) is emitted second.
// CHECK: aie.core(%[[WRITER]])
// CHECK: aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK: vector.transfer_write {{.*}}, %[[SHARED_BUF]]
// CHECK: aie.use_lock(%[[CONS_LOCK]], Release, 1)

module {
  func.func @single_herd_shared_l1() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @segment_0 {
        // Shared L1 buffer allocated at segment level, passed to the single herd.
        %shared = memref.alloc() : memref<64xbf16, 2>
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2_0) args(%sbuf=%shared) : memref<64xbf16, 2> attributes {x_loc = 0, y_loc = 2} {
          %c0 = arith.constant 0 : index
          scf.index_switch %ty
          case 0 {
            %cst = arith.constant 1.0 : bf16
            %v = vector.broadcast %cst : bf16 to vector<16xbf16>
            vector.transfer_write %v, %sbuf[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<64xbf16, 2>
            scf.yield
          }
          default {
            %cst0 = arith.constant 0.0 : bf16
            %v = vector.transfer_read %sbuf[%c0], %cst0 {in_bounds = [true]} : memref<64xbf16, 2>, vector<16xbf16>
            scf.yield
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
