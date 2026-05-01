//===- air_channel_mmio_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Negative tests for channel_type="mmio". Each split runs under `not`
// so FileCheck sees only that split's diagnostic.

// RUN: not air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu1" 2>&1 | FileCheck %s

// blockwrite encodes data directly in the instruction stream, so the
// put source must be a compile-time constant.
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

// V1: source memref.global must have no users outside the put's func,
// or the post-airrt-to-npu `symbol-dce` (tools/aircc/aircc.cpp) can't
// drop the module-level original → llvm.mlir.global collision.
// CHECK: channel_type="mmio" V1 requires the source memref.global to be used only inside the func containing the put
memref.global "private" @shared_const : memref<8xi32> = dense<3>
air.channel @sc_chan [] {channel_type = "mmio"}
func.func @other_user() -> memref<8xi32> {
  %g = memref.get_global @shared_const : memref<8xi32>
  return %g : memref<8xi32>
}
func.func @mmio_shared_global() {
  %src = memref.get_global @shared_const : memref<8xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<8xi32> {
    air.channel.put @sc_chan[] (%a[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @sc_chan[] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// Sub-byte / non-byte-aligned element types have no portable raw-byte
// repack; reject up front.
// CHECK: channel_type="mmio" source element bitwidth must be a positive multiple of 8
memref.global "private" @i1_const : memref<32xi1> = dense<true>
air.channel @i1_chan [] {channel_type = "mmio"}
func.func @mmio_subbyte_elt() {
  %src = memref.get_global @i1_const : memref<32xi1>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<32xi1> {
    air.channel.put @i1_chan[] (%a[] [] []) : (memref<32xi1>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<32xi1, 2>
        air.channel.get @i1_chan[] (%alloc[] [] []) : (memref<32xi1, 2>)
        memref.dealloc %alloc : memref<32xi1, 2>
      }
    }
  }
  return
}

// -----

// blockwrite is i32-granular; payloads not a multiple of 4 bytes
// (here 3 bf16 = 6 bytes) can't be repacked to memref<Nxi32>.
// CHECK: channel_type="mmio" source size must be a multiple of 4 bytes (got 6)
memref.global "private" @bf16_unaligned : memref<3xbf16> = dense<1.5>
air.channel @bf16u_chan [] {channel_type = "mmio"}
func.func @mmio_unaligned_payload() {
  %src = memref.get_global @bf16_unaligned : memref<3xbf16>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<3xbf16> {
    air.channel.put @bf16u_chan[] (%a[] [] []) : (memref<3xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<3xbf16, 2>
        air.channel.get @bf16u_chan[] (%alloc[] [] []) : (memref<3xbf16, 2>)
        memref.dealloc %alloc : memref<3xbf16, 2>
      }
    }
  }
  return
}

// -----

// Repack needs a DenseElementsAttr initializer; a pure declaration
// (no `= dense<...>`) used to crash on the optional dereference.
// CHECK: channel_type="mmio" non-i32 source requires a DenseElementsAttr initializer on the memref.global
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
