//===- emit_txn_cpp_shared_device.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two funcs whose segments share a name compile to one device with one
// runtime sequence per func. aiecc writes a module per sequence, each holding
// every sequence of the device; --emit-txn-cpp emits one header with a
// builder per sequence.

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aircc %s --device=npu2 --tmpdir=%t --output-format=none --emit-txn-cpp
// RUN: FileCheck %s --input-file=%t/npu.emit_txn_cpp_shared_device.txn.h

// CHECK-DAG: generate_txn_seg_k(
// CHECK-DAG: generate_txn_seg_k2(

module {
  air.channel @rb [1]
  air.channel @rb2 [1]
  func.func @k(%arg0: memref<1048576xbf16>, %nblk: index) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0, %n=%nblk) : memref<1048576xbf16>, index {
      air.segment @seg args(%sa=%a, %sn=%n) : memref<1048576xbf16>, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index
        air.channel.put @rb[%c0] (%sa[%c0, %c0, %c0] [%sn, %c16, %c256] [%c4096, %c256, %c1_0]) : (memref<1048576xbf16>)
        air.herd @h tile (%x, %y) in (%sxx=%c1_0, %syy=%c1_0) {
          %alloc = memref.alloc() : memref<4096xbf16, 2 : i32>
          %cc0 = arith.constant 0 : index
          air.channel.get @rb[%cc0] (%alloc[] [] []) : (memref<4096xbf16, 2 : i32>)
          memref.dealloc %alloc : memref<4096xbf16, 2 : i32>
        }
      }
    }
    return
  }
  func.func @k2(%arg0: memref<1048576xbf16>, %nblk: index) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0, %n=%nblk) : memref<1048576xbf16>, index {
      air.segment @seg args(%sa=%a, %sn=%n) : memref<1048576xbf16>, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index
        air.channel.put @rb2[%c0] (%sa[%c0, %c0, %c0] [%sn, %c16, %c256] [%c4096, %c256, %c1_0]) : (memref<1048576xbf16>)
        air.herd @h tile (%x, %y) in (%sxx=%c1_0, %syy=%c1_0) {
          %alloc = memref.alloc() : memref<4096xbf16, 2 : i32>
          %cc0 = arith.constant 0 : index
          air.channel.get @rb2[%cc0] (%alloc[] [] []) : (memref<4096xbf16, 2 : i32>)
          memref.dealloc %alloc : memref<4096xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
