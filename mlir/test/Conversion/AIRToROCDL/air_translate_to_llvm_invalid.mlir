//===- air_translate_to_llvm_invalid.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt --air-translate-to-llvm -verify-diagnostics %s

func.func @translate_dynamic_offset(%src : memref<16x8xf32, strided<[64, 1], offset: ?>>, %from : index, %to : index, %bases : memref<?xindex>) -> memref<16x8xf32, strided<[64, 1], offset: ?>> {
  // expected-error @+1 {{requires static strides and offset on the source}}
  %peer = air.translate %src, %from, %to, %bases : memref<16x8xf32, strided<[64, 1], offset: ?>>, memref<?xindex>
  return %peer : memref<16x8xf32, strided<[64, 1], offset: ?>>
}
