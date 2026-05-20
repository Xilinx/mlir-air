//===- opt_shim_dma_bds_per_launch_attr_invalid.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Invalid `air.shim_dma_tile_sizes`: empty, non-skip negatives, and
// multi-value 0 must produce a pass-failure diagnostic.

// RUN: not air-opt %s -split-input-file -air-opt-shim-dma-bds="device=npu2" \
// RUN:   2>&1 | FileCheck %s

air.channel @c [1]

// CHECK: air.shim_dma_tile_sizes must not be empty
func.func @empty_attr(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1)
      args(%a=%arg0) : memref<512xbf16>
      attributes {air.shim_dma_tile_sizes = array<i64>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %tok0 = air.wait_all async
    %1 = scf.for %j = %c0 to %c8 step %c1_0
        iter_args(%tok = %tok0) -> (!air.async.token) {
      %2 = air.channel.put async [%tok] @c[]
          (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
          {metadata = @airMemcpy} : (memref<512xbf16>)
      scf.yield %2 : !air.async.token
    }
  }
  return
}

// -----

air.channel @c [1]

// CHECK: air.shim_dma_tile_sizes values must be > 0
func.func @multivalue_zero(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1)
      args(%a=%arg0) : memref<512xbf16>
      attributes {air.shim_dma_tile_sizes = array<i64: 2, 0>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %tok0 = air.wait_all async
    %1 = scf.for %j = %c0 to %c8 step %c1_0
        iter_args(%tok = %tok0) -> (!air.async.token) {
      %2 = air.channel.put async [%tok] @c[]
          (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
          {metadata = @airMemcpy} : (memref<512xbf16>)
      scf.yield %2 : !air.async.token
    }
  }
  return
}

// -----

air.channel @c [1]

// CHECK: air.shim_dma_tile_sizes values must be > 0
func.func @negative_value(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%i) in (%n=%c1)
      args(%a=%arg0) : memref<512xbf16>
      attributes {air.shim_dma_tile_sizes = array<i64: -1>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c1_0 = arith.constant 1 : index
    %tok0 = air.wait_all async
    %1 = scf.for %j = %c0 to %c8 step %c1_0
        iter_args(%tok = %tok0) -> (!air.async.token) {
      %2 = air.channel.put async [%tok] @c[]
          (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
          {metadata = @airMemcpy} : (memref<512xbf16>)
      scf.yield %2 : !air.async.token
    }
  }
  return
}
