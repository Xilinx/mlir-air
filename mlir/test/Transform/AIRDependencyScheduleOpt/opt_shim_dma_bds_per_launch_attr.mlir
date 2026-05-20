//===- opt_shim_dma_bds_per_launch_attr.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Per-launch `air.shim_dma_tile_sizes` attribute decision flow in
// `air-opt-shim-dma-bds`. With CLI `shim-dma-tile-sizes` empty, the pass
// reads the per-launch attribute if present:
//   array<i64: 0>  -> sentinel, skip tiling for this launch only
//                     (per-launch wait_all fallback ties off ALL channel
//                     tokens of the launch)
//   array<i64: N>  -> auto-expand to launch's perfectly-nested depth
//                     (tilePerfectlyNested runs; wait_all ties off the
//                     single tail token)
// Absent attribute keeps the default (tile every level by 1).
// CLI override wins over per-launch attribute.
//
// Loop pattern: cascade-shape (two distinct BD patterns per iter on the
// same channel -- wrap-stride fold cannot collapse them into a single
// BD because they reference different memrefs). Trip count = 8. All
// configurations end up with 16 channel.puts (the wrap-stride fold
// normalises BD counts), but the post-pass wait_all shape differs:
//   skip-tile path -> wait_all gathers ALL channel tokens
//   tile path      -> wait_all gathers only the tail token

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" \
// RUN:   | FileCheck %s --check-prefix=ATTR
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2 shim-dma-tile-sizes=4" \
// RUN:   | FileCheck %s --check-prefix=CLI

module {
  air.channel @c_skip [1]
  air.channel @c_expand [1]
  air.channel @c_default [1]

  // Attribute `array<i64: 0>` triggers skip-tile path. Per-launch wait_all
  // fallback gathers ALL channel tokens (multi-operand list).
  // CLI=4 override forces tile path; wait_all then carries only tail token.

  // ATTR-LABEL: func.func @skip_via_attr
  // ATTR: air.shim_dma_tile_sizes = array<i64: 0>
  // ATTR: air.wait_all [%{{[0-9]+}}, %{{[0-9]+}}{{.*}}air.launch_end

  // CLI-LABEL: func.func @skip_via_attr
  // CLI: air.wait_all [%{{[0-9]+}}]  {air.launch_end}
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

  // Attribute `array<i64: 2>` triggers tile path on the single
  // perfectly-nested loop. Wait_all carries only the tail token.

  // ATTR-LABEL: func.func @autoexpand_via_attr
  // ATTR: air.shim_dma_tile_sizes = array<i64: 2>
  // ATTR: air.wait_all [%{{[0-9]+}}]  {air.launch_end}

  // CLI-LABEL: func.func @autoexpand_via_attr
  // CLI: air.wait_all [%{{[0-9]+}}]  {air.launch_end}
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

  // No attribute -> pre-existing default (tile=1) still fires.

  // ATTR-LABEL: func.func @default_no_attr
  // ATTR: air.wait_all [%{{[0-9]+}}]  {air.launch_end}

  // CLI-LABEL: func.func @default_no_attr
  // CLI: air.wait_all [%{{[0-9]+}}]  {air.launch_end}
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
}
