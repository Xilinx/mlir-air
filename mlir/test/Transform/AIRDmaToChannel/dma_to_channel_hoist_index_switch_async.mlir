//===- dma_to_channel_hoist_index_switch_async.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// Already async input, where a switch in a loop has no results. Its hoisted
// copy gets a token, and the hoisted loop waits on it.

// CHECK-LABEL: func.func @resultless_in_loop
// CHECK: scf.for {{.*}} iter_args
// CHECK: %[[S:.*]] = scf.index_switch %{{.*}} -> !air.async.token
// CHECK: air.wait_all async [{{.*}}%[[S]]{{[],]}}
// CHECK: scf.yield
// CHECK: air.segment

module {
  air.channel @chan [1, 1]
  func.func @resultless_in_loop(%a: memref<4x16xi32>, %b: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%la=%a, %lb=%b) : memref<4x16xi32>, memref<4x16xi32> {
      %1 = air.segment @segr async args(%sa=%la, %sb=%lb) : memref<4x16xi32>, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %t0 = air.wait_all async
        %l = scf.for %i = %c0 to %c2 step %c1_s iter_args(%t = %t0) -> (!air.async.token) {
          scf.index_switch %i
          case 0 {
            %g = air.channel.get async [%t] @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
            scf.yield
          }
          default {
            %g = air.channel.get async [%t] @chan[] (%sb[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
            scf.yield
          }
          %w = air.wait_all async [%t]
          scf.yield %w : !air.async.token
        }
        %2 = air.herd @hr async tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %t1 = air.wait_all async
          %hl = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%u = %t1) -> (!air.async.token) {
            %at, %l1 = air.execute [%u] -> (memref<16xi32, 2>) {
              %m = memref.alloc() : memref<16xi32, 2>
              air.execute_terminator %m : memref<16xi32, 2>
            }
            %p = air.channel.put async [%at] @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
            %d = air.execute [%p] {
              memref.dealloc %l1 : memref<16xi32, 2>
            }
            scf.yield %d : !air.async.token
          }
        }
      }
    }
    return
  }
}
