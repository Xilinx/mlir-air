//===- dma_to_channel_hoist_index_switch_error.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -split-input-file -verify-diagnostics
// RUN: not air-opt %s -air-dma-to-channel -split-input-file 2>&1 | FileCheck %s

// A switch around a host transfer cannot be copied to the launch with it when
// it also yields a value, or when its selector reads memory the host cannot
// see. It is rejected, and the pass fails, rather than the transfer dropped.

// CHECK-COUNT-2: error: 'scf.index_switch' op cannot be hoisted with the transfers it holds

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

// -----

module {
  air.channel @chan [1, 1]
  func.func @selector_reads_memory(%a: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a) : memref<4x16xi32> {
      %1 = air.segment @seg async args(%sa=%la) : memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %ta, %buf = air.execute -> (memref<1xi32, 1>) {
          %m = memref.alloc() : memref<1xi32, 1>
          air.execute_terminator %m : memref<1xi32, 1>
        }
        %tl, %v = air.execute [%ta] -> (i32) {
          %l = memref.load %buf[%c0] : memref<1xi32, 1>
          air.execute_terminator %l : i32
        }
        %key = arith.index_cast %v : i32 to index
        // expected-error @+1 {{cannot be hoisted with the transfers it holds}}
        %t = scf.index_switch %key -> !air.async.token
        case 0 {
          %g = air.channel.get async [%tl] @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield %g : !air.async.token
        }
        default {
          %e = air.wait_all async [%tl]
          scf.yield %e : !air.async.token
        }
        %2 = air.herd @h async [%t] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
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
