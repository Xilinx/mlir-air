//===- l2_memtile_column_pin_invalid.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air.memtile_col is a hint: a column with no memtile in the target warns and
// falls through to the default assignment rather than failing. The buffer is
// still placed on some memtile.

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=5 device=xcve2802 use-objectfifo=false" -verify-diagnostics | FileCheck %s

// CHECK: aie.buffer({{.*}}) {{{.*}}} : memref<32xi32, 1>

module {
  air.channel @ch_a [1, 1]
  func.func @invalid_pin() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 attributes {x_size = 3 : i64} {
        %c1_0 = arith.constant 1 : index
        %async_token_0, %alloc_0 = air.execute -> (memref<32xi32, 1>) {
          // expected-warning @+1 {{air.memtile_col=999 has no memtile in target; ignoring}}
          %a = memref.alloc() {air.memtile_col = 999 : i32} : memref<32xi32, 1>
          air.execute_terminator %a : memref<32xi32, 1>
        }
        %t0 = air.channel.put async [%async_token_0] @ch_a[] (%alloc_0[] [] []) : (memref<32xi32, 1>)
        %h0 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) attributes {x_loc = 6 : i64, y_loc = 3 : i64} {
          %async_token_10, %l1 = air.execute -> (memref<32xi32, 2>) {
            %b = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %b : memref<32xi32, 2>
          }
          %g = air.channel.get async [%async_token_10] @ch_a[] (%l1[] [] []) : (memref<32xi32, 2>)
          %d = air.execute [%g] { memref.dealloc %l1 : memref<32xi32, 2> }
        }
      }
    }
    return
  }
}
