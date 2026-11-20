//===- branch_yields_token_error.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -verify-diagnostics

// A branch that already yields an async token but has other async work in it
// is rejected: the token the pass adds would follow the yielded one, and
// dependency users would take the first.

module {
  air.channel @chan [1, 1]
  func.func @if_token(%a: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a) : memref<4x16xi32> {
      air.segment @seg args(%w=%lx, %sa=%la) : index, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %cond = arith.cmpi eq, %w, %c0 : index
        // expected-error @+1 {{already yields an async token}}
        %t = scf.if %cond -> !air.async.token {
          %g = air.channel.get async @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          %d = air.channel.get async @chan[] (%sa[%c1_s, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield %g : !air.async.token
        } else {
          %e = air.wait_all async
          scf.yield %e : !air.async.token
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) {
          %l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.put @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %l1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
}
