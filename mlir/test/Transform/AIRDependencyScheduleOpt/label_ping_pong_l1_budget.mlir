//===- label_ping_pong_l1_budget.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Labeling a loop for ping-pong doubles every L1 buffer its region tree
// allocates. When the herd tile cannot hold the result, the design used to
// fail far downstream with nothing pointing back at the labeler -- which is
// why the bfp16 matmuls carried air.disable_ping_pong by hand. With a device
// named, the labeler declines instead.
//
// The budget is the target model's local memory: 64 KiB on npu2/AIE2P.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong="device=npu2" | FileCheck %s --check-prefix=BUDGET
// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s --check-prefix=NODEV

// =============================================================================
// Case 1 (NEGATIVE under the budget): a 40 KiB L1 alloc. Doubling it needs
// 80 KiB against a 64 KiB tile, so the candidate must be declined.
//
// Without a device the check is off and the SAME loop is labeled -- that pair
// is what proves the budget is doing the work here, not some other predicate.
// =============================================================================

// BUDGET-LABEL: func.func @over_budget_declines
// BUDGET:       scf.for
// BUDGET-NOT:   hoist_alloc
// BUDGET-NOT:   } {unroll
// BUDGET:       return

// NODEV-LABEL: func.func @over_budget_declines
// NODEV:       memref.alloc() {hoist_alloc = true} : memref<160x128xbf16, 2>
// NODEV:       } {unroll = 2 : i32}

module {
  air.channel @load_chan [1]
  func.func @over_budget_declines(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          // 160*128*2 = 40960 B. 2x = 81920 B > 65536 B.
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<160x128xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<160x128xbf16, 2>
              air.execute_terminator %alloc_a : memref<160x128xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<160x128xbf16, 2>)
            %async_token_d = air.execute [%fill] {
              memref.dealloc %results_a : memref<160x128xbf16, 2>
            }
            scf.yield %async_token_d : !air.async.token
          }
        }
      }
    }
    return
  }

// =============================================================================
// Case 2 (POSITIVE control): same shape of body, 2 KiB alloc. 4 KiB fits, so
// the budget must not reject it. Locks the check against rejecting everything.
// =============================================================================

// BUDGET-LABEL: func.func @under_budget_labels
// BUDGET:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// BUDGET:       } {unroll = 2 : i32}

  func.func @under_budget_labels(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 2 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_1 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<32x32xbf16, 2>)
            %async_token_d = air.execute [%fill] {
              memref.dealloc %results_a : memref<32x32xbf16, 2>
            }
            scf.yield %async_token_d : !air.async.token
          }
        }
      }
    }
    return
  }

// =============================================================================
// Case 3 (NEGATIVE, co-tenant): the candidate's own buffer is small enough to
// double, but a sibling L1 alloc already resident on the tile is not counted
// by the candidate alone. 24 + 24 + 24 = 72 KiB. The budget must count the
// herd body's whole footprint, not just what it duplicates.
// =============================================================================

// BUDGET-LABEL: func.func @cotenant_pushes_over
// BUDGET:       scf.for
// BUDGET-NOT:   hoist_alloc
// BUDGET-NOT:   } {unroll
// BUDGET:       return

  func.func @cotenant_pushes_over(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 3 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_2 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          // Resident co-tenant: 96*128*2 = 24576 B.
          %async_token_r, %results_r = air.execute -> (memref<96x128xbf16, 2>) {
            %alloc_r = memref.alloc() : memref<96x128xbf16, 2>
            air.execute_terminator %alloc_r : memref<96x128xbf16, 2>
          }
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_r) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<96x128xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<96x128xbf16, 2>
              air.execute_terminator %alloc_a : memref<96x128xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<96x128xbf16, 2>)
            %async_token_d = air.execute [%fill] {
              memref.dealloc %results_a : memref<96x128xbf16, 2>
            }
            scf.yield %async_token_d : !air.async.token
          }
          %async_token_rd = air.execute [%3] {
            memref.dealloc %results_r : memref<96x128xbf16, 2>
          }
        }
      }
    }
    return
  }
}
