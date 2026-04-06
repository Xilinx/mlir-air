//===- air_channel_iter_count_merge.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Phase 2d iter_count inference:
//   When --air-hierarchy-to-aie unrolls air.channel @Chan[2,1] into 2 scalar
//   channels (Chan_0, Chan_1) that both map to the same aie.external_buffer,
//   Pass B Phase 2d merges them into a single canonical conduit.create and
//   must set iter_count = 2 so Pass C emits a linear 2-entry BD chain
//   (one S2MM fires twice) instead of a 1-entry circular ring.
//
// The test input is a minimal post-hierarchy aie.device IR:
//   - aie.tile(2, 1): MemTile (row=1)
//   - aie.external_buffer %ext: shared by Chan_0 and Chan_1 puts (no mem space
//     → shim inference; same SSA value triggers merge)
//   - air.channel @Chan_0 [1,1] and @Chan_1 [1,1]: two scalar declarations
//   - Two device-body-level air.channel.put ops (shim row=0): each references
//     the same %ext value, triggering Phase 2d grouping
//   - aie.core(2,1) with two air.channel.get ops (L2 memref, space 1): each
//     pulls from Chan_0 / Chan_1 respectively

// RUN: air-opt --air-channel-to-conduit %s | FileCheck %s

// The two scalar channels collapse to a single canonical conduit.create.
// CHECK:     conduit.create @Chan_0
// CHECK-SAME: iter_count = 2

// The non-canonical Chan_1 must be erased (no conduit.create for it).
// CHECK-NOT: conduit.create @Chan_1

// Both put_memref_async ops are redirected to the canonical @Chan_0.
// CHECK-DAG: conduit.put_memref_async @Chan_0
// CHECK-DAG: conduit.get_memref_async @Chan_0

// No residual air.channel ops.
// CHECK-NOT: air.channel.put
// CHECK-NOT: air.channel.get

module {
  aie.device(npu2) {
    // MemTile at (col=2, row=1) — required so inferNonCoreTile finds a MemTile.
    %tile_2_1 = aie.tile(2, 1)

    // External buffer at L3 (no memory space) → put ops using this buffer
    // will be classified as shim (row=0) by inferNonCoreTile.
    %ext = aie.external_buffer {sym_name = "ext_buf"} : memref<128xi32>

    // Two scalar air.channel declarations that Phase 2d will merge.
    air.channel @Chan_0 [1, 1]
    air.channel @Chan_1 [1, 1]

    // Device-body-level puts: outside aie.core, referencing the same %ext SSA
    // value.  inferNonCoreTile sees no memory space → classifies as shim
    // (row=0).  Both map to the same external buffer → same extBufGroups entry.
    air.channel.put @Chan_0[] (%ext[] [] []) : (memref<128xi32>)
    air.channel.put @Chan_1[] (%ext[] [] []) : (memref<128xi32>)

    // MemTile core with two gets — one for each scalar channel.  Memory space
    // 1 → classified as MemTile (row=1) by inferNonCoreTile.
    %core_2_1 = aie.core(%tile_2_1) {
      %alloc_a = memref.alloc() : memref<128xi32, 1 : i32>
      %alloc_b = memref.alloc() : memref<128xi32, 1 : i32>
      air.channel.get @Chan_0[] (%alloc_a[] [] []) : (memref<128xi32, 1 : i32>)
      air.channel.get @Chan_1[] (%alloc_b[] [] []) : (memref<128xi32, 1 : i32>)
      memref.dealloc %alloc_a : memref<128xi32, 1 : i32>
      memref.dealloc %alloc_b : memref<128xi32, 1 : i32>
      aie.end
    }
  }
}
