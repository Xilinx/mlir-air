//===- shared_l1_asymmetric_scope.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// Regression: AIR's `allocateSharedL1BufferLocks` picks per-core lock scope
// independently (outermost scf.for ancestoring all accesses, AIRToAIEPass.cpp
// :1176-1193). When producer and consumer access the shared buffer at
// DIFFERENT loop nesting depths, the acquire/release cadence mismatches:
// producer at function scope releases cons_lock once per AIE-core iter,
// consumer inside an scf.for acquires cons_lock N times per iter ⇒
// permanent stall after iter 1 (consumer drains the single token then
// blocks waiting for more, producer is blocked outside the loop).
//
// An N-writer + 1-reader shared-L1 design avoids this
// because its kernel internally manages locks via lock_acquire/release
// intrinsics; AIR doesn't emit acquire/release wrappers at all.
//
// Fix in allocateSharedL1BufferLocks: enforce CROSS-CORE scope symmetry.
// When the per-core lockScopes are at different nesting levels (or one
// is NULL = function scope), HOIST all to the deepest common
// core-body-level statement. For each accessing op, wrap the
// core-body-level statement containing it (NOT the op itself) with
// acquire/release. This produces one acquire-release per AIE-core iter
// regardless of inner scf.for nesting.

// CHECK: aie.device

// Producer at tile_0_2, consumer at tile_0_3 (adjacent rows).
// CHECK-DAG: %[[PROD_TILE:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[CONS_TILE:.*]] = aie.tile(0, 3)

// Producer lock init=1, consumer lock init=0 (1-producer 1-consumer).
// CHECK-DAG: %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 1 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}

// CONSUMER core body: acquire MUST happen BEFORE the scf.for loop
// (hoisted out), not inside each iteration. Release MUST happen AFTER
// the scf.for. This avoids the cadence mismatch.
// CHECK: aie.core(%[[CONS_TILE]])
// CHECK: aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK: scf.for
// Negative: no acquire/release INSIDE the consumer's scf.for body.
// (Pre-fix the inner-scope choice puts both lock ops inside the scf.for.)
// CHECK-NOT: aie.use_lock
// CHECK: }
// CHECK: aie.use_lock(%[[PROD_LOCK]], Release, 1)

// CHECK: aie.core(%[[PROD_TILE]])

module {
  func.func @shared_l1_asym_scope(%out: memref<64xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%out_arg=%out) : memref<64xbf16> {
      air.segment @segment_0 args(%out_seg=%out_arg) : memref<64xbf16> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c8_s = arith.constant 8 : index
        // Shared L1 buffer at segment scope.
        %alloc_shared = memref.alloc() : memref<8xbf16, 2>

        // PRODUCER (tile_0_2): writes to shared buf at FUNCTION SCOPE
        // (NO inner scf.for around the write).
        air.herd @herd_producer tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s)
            args(%shared_buf=%alloc_shared) : memref<8xbf16, 2>
            attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %cst = arith.constant 1.0 : bf16
          %vec = vector.broadcast %cst : bf16 to vector<8xbf16>
          %c0_h = arith.constant 0 : index
          // Write at function scope (no inner scf.for).
          vector.transfer_write %vec, %shared_buf[%c0_h] {in_bounds = [true]}
              : vector<8xbf16>, memref<8xbf16, 2>
        }

        // CONSUMER (tile_0_3): reads shared buf INSIDE an scf.for
        // (asymmetric scope vs producer). With the bug, AIR placed
        // acquire INSIDE this scf.for → 8 acquires per AIE-core iter
        // while producer only releases 1 → deadlock.
        air.herd @herd_consumer tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s)
            args(%shared_buf=%alloc_shared) : memref<8xbf16, 2>
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c8_h = arith.constant 8 : index
          %local = memref.alloc() : memref<8xbf16, 2>
          scf.for %i = %c0_h to %c8_h step %c1_h {
            %v = memref.load %shared_buf[%i] : memref<8xbf16, 2>
            memref.store %v, %local[%i] : memref<8xbf16, 2>
          }
          memref.dealloc %local : memref<8xbf16, 2>
        }
      }
    }
    return
  }
}
