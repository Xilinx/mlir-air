//===- air_channel_iter_count_merge.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Phase 2d shim relay channel merge:
//   When --air-hierarchy-to-aie unrolls air.channel @Chan[2,1] into 2 scalar
//   channels (Chan_0, Chan_1) that both map to the same aie.external_buffer,
//   Pass B Phase 2d merges them into a single canonical conduit.create.
//   The surviving channel accumulates all put/get ops, which Pass C lowers
//   into a BD chain on a single physical DMA channel.
//
// The test input is a minimal post-hierarchy aie.device IR:
//   - aie.tile(2, 1): MemTile (row=1)
//   - aie.external_buffer %ext: shared by Chan_0 and Chan_1 puts (no mem space
//     -> shim inference; same SSA value triggers merge)
//   - air.channel @Chan_0 [1,1] and @Chan_1 [1,1]: two scalar declarations
//   - Two device-body-level air.channel.put ops (shim row=0): each references
//     the same %ext value, triggering Phase 2d grouping
//   - Two device-body-level air.channel.get ops (L2 memref, space 1):
//     inferNonCoreTile classifies them as MemTile (row=1) consumers

// RUN: air-opt --air-channel-to-conduit %s 2>&1 | FileCheck %s

// Phase 2d emits a merge remark.
// CHECK: remark{{.*}}merged shim relay channels

// CHECK-LABEL: module

// The two scalar channels collapse to a single canonical conduit.create.
// CHECK:     conduit.create @Chan_0
// CHECK-SAME: element_type = memref<128xi32, 1 : i32>

// The non-canonical Chan_1 must be erased (no conduit.create for it).
// CHECK-NOT: conduit.create @Chan_1

// Both put_memref_async ops are redirected to the canonical @Chan_0.
// CHECK:     conduit.put_memref_async
// CHECK-SAME: name = @Chan_0
// CHECK:     conduit.put_memref_async
// CHECK-SAME: name = @Chan_0

// Both get_memref_async ops are redirected to the canonical @Chan_0.
// CHECK:     conduit.get_memref_async
// CHECK-SAME: name = @Chan_0
// CHECK:     conduit.get_memref_async
// CHECK-SAME: name = @Chan_0

// No residual air.channel ops.
// CHECK-NOT: air.channel.put
// CHECK-NOT: air.channel.get

module {
  aie.device(npu2) {
    // MemTile at (col=2, row=1) — required so inferNonCoreTile finds a MemTile.
    %tile_2_1 = aie.tile(2, 1)

    // External buffer at L3 (no memory space) -> put ops using this buffer
    // will be classified as shim (row=0) by inferNonCoreTile.
    %ext = aie.external_buffer {sym_name = "ext_buf"} : memref<128xi32>

    // Two scalar air.channel declarations that Phase 2d will merge.
    air.channel @Chan_0 [1, 1]
    air.channel @Chan_1 [1, 1]

    // Device-body-level puts: outside aie.core, referencing the same %ext SSA
    // value.  inferNonCoreTile sees no memory space -> classifies as shim
    // (row=0).  Both map to the same external buffer -> same extBufGroups entry.
    air.channel.put @Chan_0[] (%ext[] [] []) : (memref<128xi32>)
    air.channel.put @Chan_1[] (%ext[] [] []) : (memref<128xi32>)

    // Device-body-level gets: memory space 1 -> inferNonCoreTile classifies
    // as MemTile (row=1) consumer.  Phase 2d requires consumer at row 1.
    %alloc_a = memref.alloc() : memref<128xi32, 1 : i32>
    %alloc_b = memref.alloc() : memref<128xi32, 1 : i32>
    air.channel.get @Chan_0[] (%alloc_a[] [] []) : (memref<128xi32, 1 : i32>)
    air.channel.get @Chan_1[] (%alloc_b[] [] []) : (memref<128xi32, 1 : i32>)
  }
}
