//===- air_channel_mmio_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Negative test: lowering rejects mmio puts whose source is not a
// memref.get_global. The runtime-sequence blockwrite encodes the data
// directly in the instruction stream, so it must be a compile-time
// constant; live host data must reach a tile via shim DMA instead.

// RUN: not air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" 2>&1 | FileCheck %s

// CHECK: channel_type="mmio" put requires source memref defined by memref.get_global

air.channel @mmio_nc [] {channel_type = "mmio"}
func.func @mmio_nonconst(%h: memref<8xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %h) : memref<8xi32> {
    air.channel.put @mmio_nc[] (%a[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @mmio_nc[] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}
