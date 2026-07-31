//===- single_herd_shared_l1_opaque_call.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Intra-herd (core-to-core) shared L1 buffer inferred across an OPAQUE
// external kernel call. A single air.herd has two cores that communicate
// through a segment-scope L1 buffer passed as a herd operand. The PRODUCER
// core writes the buffer via a func.call to an external kernel that carries
// no MemoryEffectOpInterface, so the memory-effect-based reader/writer scan
// cannot see the write. The producer role is selected from the tile index
// (scf.index_switch on %ty), so the call is reachable on a strict subset of
// the herd's cores (one of two). air::herdBufferHasCrossCoreDependence must
// still recognize the cross-core producer/consumer dependence from the opaque
// write -- otherwise the buffer is wrongly sunk into each core as a private
// copy (two local allocs, no shared aie.buffer) and the neighbor-L1 hand-off
// breaks. (A call reachable on EVERY core is replication, not communication,
// and must NOT be flagged -- see Transform/AIRVerifyHierarchyLocality tests.)
//
// Expected: ONE shared aie.buffer (not a per-core clone) with a producer/
// consumer lock pair, referenced by BOTH cores.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir' | FileCheck %s

// The buffer becomes a single shared aie.buffer with a producer/consumer lock
// pair. If the opaque write were missed, the buffer would instead be cloned as
// a per-core memref.alloc and NO "shared_l1" symbol would be emitted at all.
// CHECK-DAG: %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 1 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CHECK-DAG: %[[SHARED_BUF:.*]] = aie.buffer(%{{.*}}) {{.*}}sym_name = "shared_l1{{.*}}"{{.*}} : memref<64xbf16, 2 : i32>

// Both cores reference the SAME shared buffer + locks (reader emitted first).
// CHECK: aie.core
// CHECK: aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual
// CHECK: vector.transfer_read %[[SHARED_BUF]]
// CHECK: aie.use_lock(%[[PROD_LOCK]], Release
// CHECK: aie.core
// CHECK: aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual
// CHECK: func.call @ext_write(%[[SHARED_BUF]])
// CHECK: aie.use_lock(%[[CONS_LOCK]], Release

module {
  func.func private @ext_write(memref<64xbf16, 2 : i32>) attributes {llvm.emit_c_interface}
  func.func @single_herd_shared_l1_opaque_call() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %shared = memref.alloc() : memref<64xbf16, 2 : i32>
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2_0) args(%sbuf=%shared) : memref<64xbf16, 2 : i32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          // ty==0 -> producer: writes the shared buffer via an OPAQUE external
          // kernel call (no visible memory effect). ty==1 -> consumer: reads it.
          scf.index_switch %ty
          case 0 {
            func.call @ext_write(%sbuf) : (memref<64xbf16, 2 : i32>) -> ()
            scf.yield
          }
          default {
            %cst0 = arith.constant 0.0 : bf16
            %v = vector.transfer_read %sbuf[%c0], %cst0 {in_bounds = [true]} : memref<64xbf16, 2 : i32>, vector<16xbf16>
            scf.yield
          }
        }
      }
    }
    return
  }
}
