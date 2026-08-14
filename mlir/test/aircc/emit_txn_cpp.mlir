//===- emit_txn_cpp.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A transfer whose extent is a function argument -- a context length known only
// at dispatch -- cannot be frozen into an insts.bin. --emit-txn-cpp emits a
// builder that assembles the instruction stream from that argument instead.

// RUN: rm -rf %t && mkdir -p %t
// RUN: aircc %s --device=npu2 --tmpdir=%t --output-format=none --emit-txn-cpp -v 2>&1 || true
// RUN: FileCheck %s --input-file=%t/npu.emit_txn_cpp.txn.h

// The builder takes the runtime extent, and that argument has to be multiplied
// into the shim BD's buffer length (register 0x1d000 = 118784) rather than
// reduced to a predicate on the static size -- which is what the DMA-task
// lowering does, silently keeping its own compile-time extent.
// CHECK: generate_txn_k_0_k(size_t v1) {
// CHECK: [[P:v[0-9]+]] = (ptrdiff_t) v1
// CHECK: [[I:v[0-9]+]] = (int64_t) [[P]]
// CHECK: [[U:v[0-9]+]] = (uint64_t) [[I]]
// CHECK: [[U32:v[0-9]+]] = (uint32_t) [[U]]
// CHECK: [[N:v[0-9]+]] = (int32_t) [[U32]]
// CHECK: [[NU:v[0-9]+]] = (uint32_t) [[N]]
// CHECK: [[LEN:v[0-9]+]] = v{{[0-9]+}} * [[NU]]
// CHECK: [[LENI:v[0-9]+]] = (int32_t) [[LEN]]
// CHECK: [[ADDR:v[0-9]+]] = 118784u
// CHECK: txn_append_write32(txn, [[ADDR]], [[LENI]])

module {
  air.channel @rb [1]
  func.func @k(%arg0: memref<1048576xbf16>, %nblk: index) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a=%arg0, %n=%nblk) : memref<1048576xbf16>, index {
      air.segment args(%sa=%a, %sn=%n) : memref<1048576xbf16>, index {
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
}
