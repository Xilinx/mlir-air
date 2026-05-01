//===- air_channel_mmio_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Negative tests for channel_type="mmio". Each split runs under `not`
// so FileCheck sees only that split's diagnostic.

// RUN: not air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu1" 2>&1 | FileCheck %s

// The source data is stamped onto the destination L1 buffer's
// initial_value, so the put source must be a compile-time constant
// memref.global.
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

// -----

// Non-broadcast mmio with non-constant index can't match any get;
// would silently erase the put. Reject up front.
// CHECK: channel_type="mmio" non-broadcast put requires compile-time constant indices
memref.global "private" @nci_const : memref<8xi32> = dense<1>
air.channel @nci_chan [1] {channel_type = "mmio"}
func.func @mmio_nonconst_index(%n: index) {
  %src = memref.get_global @nci_const : memref<8xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src, %k = %n) : memref<8xi32>, index {
    air.channel.put @nci_chan[%k] (%a[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @nci_chan[%c0] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// Constant-index put with no matching get would be silently erased.
// CHECK: channel_type="mmio" put has no matching device-side air.channel.get
memref.global "private" @nm_const : memref<8xi32> = dense<2>
air.channel @nm_chan [2] {channel_type = "mmio"}
func.func @mmio_no_match() {
  %src = memref.get_global @nm_const : memref<8xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<8xi32> {
    %c1_p = arith.constant 1 : index
    air.channel.put @nm_chan[%c1_p] (%a[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @nm_chan[%c0] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// The destination L1 buffer's element type must match the source so the
// initializer is type-compatible.
// CHECK: channel_type="mmio" source/destination element type mismatch
memref.global "private" @i32_src : memref<4xi32> = dense<7>
air.channel @typemis_chan [] {channel_type = "mmio"}
func.func @mmio_type_mismatch() {
  %src = memref.get_global @i32_src : memref<4xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<4xi32> {
    air.channel.put @typemis_chan[] (%a[] [] []) : (memref<4xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<4xbf16, 2>
        air.channel.get @typemis_chan[] (%alloc[] [] []) : (memref<4xbf16, 2>)
        memref.dealloc %alloc : memref<4xbf16, 2>
      }
    }
  }
  return
}

// -----

// Source/destination must agree on total element count.
// CHECK: channel_type="mmio" source/destination element count mismatch
memref.global "private" @short_src : memref<4xi32> = dense<7>
air.channel @sizemis_chan [] {channel_type = "mmio"}
func.func @mmio_size_mismatch() {
  %src = memref.get_global @short_src : memref<4xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<4xi32> {
    air.channel.put @sizemis_chan[] (%a[] [] []) : (memref<4xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @sizemis_chan[] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// initial_value is set by the lowering, so the source memref.global
// needs a DenseElementsAttr initializer to copy from.
// CHECK: channel_type="mmio" source memref.global must have a DenseElementsAttr initializer
memref.global "private" @uninit_bf16 : memref<2x2xbf16>
air.channel @uninit_chan [] {channel_type = "mmio"}
func.func @mmio_uninitialized_global() {
  %src = memref.get_global @uninit_bf16 : memref<2x2xbf16>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<2x2xbf16> {
    air.channel.put @uninit_chan[] (%a[] [] []) : (memref<2x2xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<2x2xbf16, 2>
        air.channel.get @uninit_chan[] (%alloc[] [] []) : (memref<2x2xbf16, 2>)
        memref.dealloc %alloc : memref<2x2xbf16, 2>
      }
    }
  }
  return
}
