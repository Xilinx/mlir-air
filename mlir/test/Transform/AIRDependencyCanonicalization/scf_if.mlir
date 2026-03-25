//===- scf_if.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// Test that air-dependency-canonicalize properly elevates tokens from ops
// inside scf.if branches when wiring dependencies to ops at the enclosing
// scf.for body level. Without the fix, the second channel.get would get
// a dependency on the channel.put INSIDE the scf.if branch (violating
// SSA dominance) instead of on the scf.if result.

// CHECK-LABEL: func.func @scf_if_token_elevation
// CHECK: air.segment
// CHECK: scf.for
// CHECK:   air.channel.get async
// CHECK:   %[[IF_RESULT:.*]] = scf.if
// CHECK:     air.channel.put async
// CHECK:   } else {
// CHECK:     air.channel.put async
// CHECK:   }
// The second channel.get must depend on the scf.if result, NOT on a
// channel.put defined inside the scf.if branch.
// CHECK:   air.channel.get async [{{.*}}%[[IF_RESULT]]{{.*}}]

module {
  air.channel @chan_in [2]
  air.channel @chan_out_0 [1, 1, 1]
  air.channel @chan_out_1 [1, 1, 1]
  func.func @scf_if_token_elevation(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg0, %arg6=%arg1, %arg7=%arg2) : memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16> {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      // Matching channel.put for chan_in (feed data into segment)
      air.channel.put  @chan_in[%c0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c1_0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c1_0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c1_0] (%arg5[] [] []) : (memref<64x64xbf16>)
      air.channel.put  @chan_in[%c1_0] (%arg5[] [] []) : (memref<64x64xbf16>)
      // Matching channel.get for chan_out_0 and chan_out_1
      air.channel.get  @chan_out_0[%c0, %c0, %c0] (%arg6[] [] []) : (memref<64x64xbf16>)
      air.channel.get  @chan_out_0[%c0, %c0, %c0] (%arg6[] [] []) : (memref<64x64xbf16>)
      air.channel.get  @chan_out_1[%c0, %c0, %c0] (%arg7[] [] []) : (memref<64x64xbf16>)
      air.channel.get  @chan_out_1[%c0, %c0, %c0] (%arg7[] [] []) : (memref<64x64xbf16>)
      %1 = air.segment @seg async  unroll(%arg8) in (%arg9=%c2) {
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2_2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c512 = arith.constant 512 : index
        %c64_3 = arith.constant 64 : index
        %2 = scf.for %iv = %c0_1 to %c2_2 step %c1_1 iter_args(%dep = %async_token) -> (!air.async.token) {
          %3 = air.channel.get async [%dep]  @chan_in[%arg8] (%results[] [] []) {id = 2 : i32} : (memref<64x64xbf16, 1 : i32>)
          %cond = arith.cmpi eq, %arg8, %c0_1 : index
          %4 = scf.if %cond -> (!air.async.token) {
            %7 = air.channel.put async [%3]  @chan_out_0[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 3 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %7 : !air.async.token
          } else {
            %7 = air.channel.put async [%3]  @chan_out_1[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 4 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %7 : !air.async.token
          }
          %5 = air.channel.get async [%dep, %4]  @chan_in[%arg8] (%results[] [] []) {id = 5 : i32} : (memref<64x64xbf16, 1 : i32>)
          %6 = scf.if %cond -> (!air.async.token) {
            %7 = air.channel.put async [%5]  @chan_out_0[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 6 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %7 : !air.async.token
          } else {
            %7 = air.channel.put async [%5]  @chan_out_1[%c0_1, %c0_1, %c0_1] (%results[%c0_1, %c0_1, %c0_1, %c0_1] [%c8, %c8, %c8, %c8] [%c8, %c512, %c64_3, %c1_1]) {id = 7 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %7 : !air.async.token
          }
          scf.yield %6 : !air.async.token
        }
        %async_token_4 = air.execute [%2] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        } {id = 8 : i32}
      }
    }
    return
  }
}
