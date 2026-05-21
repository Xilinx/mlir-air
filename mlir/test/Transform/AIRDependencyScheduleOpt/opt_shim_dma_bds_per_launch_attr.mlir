//===- opt_shim_dma_bds_per_launch_attr.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Per-launch `air.shim_dma_tile_sizes` decision flow in `air-opt-shim-dma-bds`:
//   [0]         skip tiling for this launch, [N] auto-expand to nest depth,
//   [a, b, ...] use literally. CLI option overrides the attribute.
// FileCheck verifies the attribute is preserved on the launch op and a
// launch-end wait_all is emitted (per-launch fallback for skip-tile,
// tiling fixup for tile-path).

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" \
// RUN:   | FileCheck %s --check-prefix=ATTR
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2 shim-dma-tile-sizes=4" \
// RUN:   | FileCheck %s --check-prefix=CLI

module {
  air.channel @c_skip [1]
  air.channel @c_expand [1]
  air.channel @c_multi [1]
  air.channel @c_default [1]
  air.channel @c_mix_a [1]
  air.channel @c_mix_b [1]

  // ATTR-LABEL: func.func @skip_via_attr
  // ATTR: air.shim_dma_tile_sizes = array<i64: 0>
  // ATTR: air.wait_all{{.*}}{air.launch_end}

  // CLI-LABEL: func.func @skip_via_attr
  // CLI: air.wait_all{{.*}}{air.launch_end}
  func.func @skip_via_attr(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16>
        attributes {air.shim_dma_tile_sizes = array<i64: 0>} {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @c_skip[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyR1} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @c_skip[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyA1} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // Attribute `array<i64: 2>` triggers the auto-expand tile path.

  // ATTR-LABEL: func.func @autoexpand_via_attr
  // ATTR: air.shim_dma_tile_sizes = array<i64: 2>
  // ATTR: air.wait_all{{.*}}{air.launch_end}

  // CLI-LABEL: func.func @autoexpand_via_attr
  // CLI: air.wait_all{{.*}}{air.launch_end}
  func.func @autoexpand_via_attr(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16>
        attributes {air.shim_dma_tile_sizes = array<i64: 2>} {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @c_expand[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyR2} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @c_expand[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyA2} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // Multi-value attribute on a 2-deep perfectly-nested band must be consumed
  // literally (no auto-expand). Verified by: attribute preserved on launch.

  // ATTR-LABEL: func.func @multivalue_via_attr
  // ATTR: air.shim_dma_tile_sizes = array<i64: 2, 2>
  // ATTR: air.wait_all{{.*}}{air.launch_end}

  // CLI-LABEL: func.func @multivalue_via_attr
  // CLI: air.wait_all{{.*}}{air.launch_end}
  func.func @multivalue_via_attr(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16>
        attributes {air.shim_dma_tile_sizes = array<i64: 2, 2>} {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c4 step %c1_0
          iter_args(%tokj = %tok0) -> (!air.async.token) {
        %2 = scf.for %k = %c0 to %c4 step %c1_0
            iter_args(%tokk = %tokj) -> (!air.async.token) {
          %3 = air.channel.put async [%tokk] @c_multi[]
              (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
              {metadata = @airMemcpyR3} : (memref<512xbf16>)
          %4 = air.channel.put async [%3] @c_multi[]
              (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
              {metadata = @airMemcpyA3} : (memref<512xbf16>)
          scf.yield %4 : !air.async.token
        }
        scf.yield %2 : !air.async.token
      }
    }
    return
  }

  // ATTR-LABEL: func.func @default_no_attr
  // ATTR: air.wait_all{{.*}}{air.launch_end}

  // CLI-LABEL: func.func @default_no_attr
  // CLI: air.wait_all{{.*}}{air.launch_end}
  func.func @default_no_attr(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @c_default[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyRdef} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @c_default[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyAdef} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // Mixed launches: one carries the skip-tile attribute, the other doesn't.
  // Verifies the per-launch attribute dispatch (the unattributed launch
  // must not pick up the attribute), and that both end up with an
  // air.launch_end wait_all.

  // ATTR-LABEL: func.func @mixed_launches
  // ATTR: air.launch
  // ATTR: air.shim_dma_tile_sizes = array<i64: 0>
  // ATTR: air.wait_all{{.*}}{air.launch_end}
  // ATTR: air.launch
  // ATTR-NOT: air.shim_dma_tile_sizes
  // ATTR: air.wait_all{{.*}}{air.launch_end}
  func.func @mixed_launches(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16>
        attributes {air.shim_dma_tile_sizes = array<i64: 0>} {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @c_mix_a[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyMixA1} : (memref<512xbf16>)
        %3 = air.channel.put async [%2] @c_mix_a[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyMixA2} : (memref<512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    %4 = air.launch async [%0] (%i) in (%n=%c1)
        args(%a=%arg0, %b=%arg1) : memref<512xbf16>, memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %5 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %6 = air.channel.put async [%tok] @c_mix_b[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyMixB1} : (memref<512xbf16>)
        %7 = air.channel.put async [%6] @c_mix_b[]
            (%b[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyMixB2} : (memref<512xbf16>)
        scf.yield %7 : !air.async.token
      }
    }
    return
  }
}
