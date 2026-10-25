//===- conditional_host_producers.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=3 col-offset=2 device=npu2" | FileCheck %s

// The host picks one of several puts with an scf.index_switch or an scf.if
// that yields their token. The device copy of the branch op keeps no token, and
// the host op keeps both puts, bound to the shim allocation.

// CHECK-LABEL: aie.device(npu2) @seg {
// CHECK-NOT: !air.async.token
// CHECK: aie.shim_dma_allocation @air_x
// CHECK: func.func @conditional_host
// CHECK: scf.index_switch
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
func.func @conditional_host(%host: memref<8xbf16>, %select: index) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%h=%host, %s=%select) : memref<8xbf16>, index {
    %choice = scf.index_switch %s -> !air.async.token
    case 0 {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 100 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    }
    default {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 101 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    }
    air.wait_all [%choice]
    air.segment @seg {
      %one = arith.constant 1 : index
      air.herd @receiver tile (%tx, %ty) in (%sx=%one, %sy=%one)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %token, %local = air.execute -> (memref<8xbf16, 2>) {
          %buffer = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %buffer : memref<8xbf16, 2>
        }
        air.channel.get @x[] (%local[] [] []) : (memref<8xbf16, 2>)
        %done = air.execute { memref.dealloc %local : memref<8xbf16, 2> }
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: aie.device(npu2) @seg {
// CHECK-NOT: !air.async.token
// CHECK: aie.shim_dma_allocation @air_x
// CHECK: func.func @conditional_host_if
// CHECK: scf.if
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
func.func @conditional_host_if(%host: memref<8xbf16>, %select: index) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%h=%host, %s=%select) : memref<8xbf16>, index {
    %c0 = arith.constant 0 : index
    %cond = arith.cmpi eq, %s, %c0 : index
    %choice = scf.if %cond -> !air.async.token {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 100 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    } else {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 101 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    }
    air.wait_all [%choice]
    air.segment @seg {
      %one = arith.constant 1 : index
      air.herd @receiver tile (%tx, %ty) in (%sx=%one, %sy=%one)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %token, %local = air.execute -> (memref<8xbf16, 2>) {
          %buffer = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %buffer : memref<8xbf16, 2>
        }
        air.channel.get @x[] (%local[] [] []) : (memref<8xbf16, 2>)
        %done = air.execute { memref.dealloc %local : memref<8xbf16, 2> }
      }
    }
  }
  return
}

// -----

// Nested switches, as when a bank select wraps a phase select.
// CHECK-LABEL: aie.device(npu2) @seg {
// CHECK-NOT: !air.async.token
// CHECK: aie.shim_dma_allocation @air_x
// CHECK: func.func @conditional_host_nested
// CHECK: scf.index_switch
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"
// CHECK: scf.index_switch
// CHECK-COUNT-2: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
func.func @conditional_host_nested(%host: memref<8xbf16>, %select: index, %bank: index) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%h=%host, %s=%select, %k=%bank) : memref<8xbf16>, index, index {
    %choice = scf.index_switch %s -> !air.async.token
    case 0 {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 100 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    }
    default {
      %inner = scf.index_switch %k -> !air.async.token
      case 0 {
        %put = air.channel.put async @x[] (%h[] [] []) {id = 101 : i32} : (memref<8xbf16>)
        scf.yield %put : !air.async.token
      }
      default {
        %put = air.channel.put async @x[] (%h[] [] []) {id = 102 : i32} : (memref<8xbf16>)
        scf.yield %put : !air.async.token
      }
      scf.yield %inner : !air.async.token
    }
    air.wait_all [%choice]
    air.segment @seg {
      %one = arith.constant 1 : index
      air.herd @receiver tile (%tx, %ty) in (%sx=%one, %sy=%one)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %token, %local = air.execute -> (memref<8xbf16, 2>) {
          %buffer = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %buffer : memref<8xbf16, 2>
        }
        air.channel.get @x[] (%local[] [] []) : (memref<8xbf16, 2>)
        %done = air.execute { memref.dealloc %local : memref<8xbf16, 2> }
      }
    }
  }
  return
}
