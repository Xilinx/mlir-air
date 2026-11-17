//===- dma_to_channel_hoist_index_switch_error.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -verify-diagnostics

// A switch around a host transfer that also yields a value cannot be copied to
// the launch with it; it is rejected rather than dropped.

module {
  air.channel @chan [1, 1]
  func.func @yields_value(%a: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a) : memref<4x16xi32> {
      %1 = air.segment @seg async args(%w=%lx, %sa=%la) : index, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        // expected-error @+1 {{cannot be hoisted with the transfers it holds}}
        %row, %t = scf.index_switch %w -> index, !air.async.token
        case 0 {
          %g = air.channel.get async @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield %c0, %g : index, !air.async.token
        }
        default {
          %e = air.wait_all async
          scf.yield %c1_s, %e : index, !air.async.token
        }
        %2 = air.herd @h async [%t] tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) args(%r=%row) : index {
          %at, %l1 = air.execute -> (memref<16xi32, 2>) {
            %m = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %m : memref<16xi32, 2>
          }
          %p = air.channel.put async [%at] @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          %d = air.execute [%p] {
            memref.dealloc %l1 : memref<16xi32, 2>
          }
        }
      }
    }
    return
  }
}
