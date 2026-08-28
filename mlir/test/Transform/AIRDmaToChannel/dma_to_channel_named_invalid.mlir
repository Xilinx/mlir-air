//===- dma_to_channel_named_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -split-input-file -verify-diagnostics

// A name with no declaration behind it is a typo. Minting a channel for it
// would turn that typo into a silent point-to-point circuit flow that only
// fails much later, in air-to-aie, with nothing left pointing back here.
func.func @undeclared_channel(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    // expected-error @+2 {{names channel @noSuchChannel, which is not declared in any enclosing symbol table}}
    // expected-error @+1 {{failed to legalize operation 'air.dma_memcpy_nd'}}
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @noSuchChannel} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// The declaration fixes the bundle rank. When it disagrees with the enclosing
// spatial iteration the correct sub-channel is genuinely unknown, so say so
// rather than guessing.
air.channel @bundle3 [3]
func.func @rank_disagrees_with_spatial(%arg0: memref<64x64xi32>) {
  %c2 = arith.constant 2 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    // expected-error @+2 {{declared with 1 bundle dimension(s), but the enclosing spatial iteration supplies 2}}
    // expected-error @+1 {{failed to legalize operation 'air.dma_memcpy_nd'}}
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @bundle3} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}
