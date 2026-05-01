//===- air_channel_mmio_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Negative tests for the channel_type="mmio" lowering. Each split is run
// individually with `not` so the FileCheck directive after it sees only
// that split's diagnostic.

// RUN: not air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu1" 2>&1 | FileCheck %s

// The runtime-sequence blockwrite encodes the data directly in the
// instruction stream, so the put source must be a compile-time constant.
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

// Non-broadcast mmio with a non-constant index would silently fail to
// match any device-side get and the put would be erased with no
// blockwrite emitted. Reject up front.
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

// Non-broadcast put whose constant index does not match any device-side
// get would otherwise be erased with no blockwrite emitted.
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

// V1 limitation: the source memref.global must have no users outside the
// func containing the put — otherwise the `symbol-dce` pass run after
// `airrt-to-npu` (in tools/aircc/aircc.cpp's NPU pipeline) can't remove
// the module-level original and a duplicate-symbol collision occurs in
// LLVM lowering.
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

// Sub-byte (e.g. i1) and other non-byte-aligned element types have no
// portable raw-byte representation and would make the (elts*bits)/8
// repack accounting lossy. Reject them up front rather than emitting
// a malformed blockwrite.
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

// blockwrite is i32-granular on the wire. A byte-aligned element type
// whose total payload size isn't a multiple of 4 bytes (here 3 bf16 = 6
// bytes) cannot be safely repacked to memref<Nxi32>; reject up front.
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

// Repack needs concrete bytes from the source memref.global. A pure
// declaration (no `= dense<...>` initializer) has none, and previously
// crashed via `std::optional::operator*` on getInitialValue(). Reject
// with a clean diagnostic.
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
