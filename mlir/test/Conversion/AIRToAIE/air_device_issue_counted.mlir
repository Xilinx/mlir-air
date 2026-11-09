//===- air_device_issue_counted.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" | FileCheck %s

// The router starts one armed put on each of two loop iterations
// (air.device_issue_count = 2). Only the first start of a run waits for the
// host's arm, and the BD address it reads back is kept: after a start the BD
// holds the patched address, not the armed one.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK-DAG: %[[TAKEN:.*]] = aie.buffer(%{{.*}}) {sym_name = "__air_issue_taken_0_2_0"} : memref<1xi32, 2 : i32>
// CHECK-DAG: %[[BASE:.*]] = aie.buffer(%{{.*}}) {sym_name = "__air_issue_base_0_2_0"} : memref<2xi32, 2 : i32>
// CHECK: aie.core
// CHECK: memref.store %{{.*}}, %[[TAKEN]]
// CHECK: scf.for
// CHECK: %[[T:.*]] = memref.load %[[TAKEN]]
// CHECK: %[[FRESH:.*]] = arith.cmpi eq, %[[T]]
// CHECK: scf.if %[[FRESH]] {
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK: arith.select %[[FRESH]]
// CHECK: memref.store %{{.*}}, %[[BASE]]
// CHECK: air.channel.put {{.*}}air.device_issue_bd = 15 : i32
module {
  air.channel @Wexp [1]
  func.func @counted(%arg0: memref<32x256xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @Wexp[%c0] (%arg6[0, 0] [1, 256] [256, 1]) {air.device_issue = 0 : i64, air.device_issue_count = 2 : i64, id = 1 : i32} : (memref<32x256xbf16>)
      %2 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %3 = air.herd @router async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c256 = arith.constant 256 : index
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %async_token, %results = air.execute -> (memref<4xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4xi32, 2 : i32>
            air.execute_terminator %alloc : memref<4xi32, 2 : i32>
          }
          %l = scf.for %i = %c0_1 to %c2 step %c1_1 iter_args(%t = %async_token) -> (!air.async.token) {
            %async_token_2, %results_3 = air.execute [%t] -> (i32) {
              %7 = memref.load %results[%i] : memref<4xi32, 2 : i32>
              air.execute_terminator %7 : i32
            }
            %5 = arith.index_cast %results_3 : i32 to index
            %6 = arith.muli %5, %c256 : index
            %async_token_5 = air.execute [%async_token_2] {
              air.channel.issue @Wexp[%c0_1] offset(%6 : index) {issue_id = 0 : i64}
            }
            scf.yield %async_token_5 : !air.async.token
          }
          %async_token_4 = air.execute [%l] {
            memref.dealloc %results : memref<4xi32, 2 : i32>
          }
        }
        %4 = air.herd @mvm async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 4 : i32, x_loc = 1 : i64, y_loc = 2 : i64} {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %async_token, %results = air.execute -> (memref<256xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<256xbf16, 2 : i32>
          }
          %l = scf.for %i = %c0_1 to %c2 step %c1_1 iter_args(%t = %async_token) -> (!air.async.token) {
            %5 = air.channel.get async [%t]  @Wexp[%c0_1] (%results[] [] []) {id = 2 : i32} : (memref<256xbf16, 2 : i32>)
            scf.yield %5 : !air.async.token
          }
          %async_token_2 = air.execute [%l] {
            memref.dealloc %results : memref<256xbf16, 2 : i32>
          }
        }
        air.wait_all [%3, %4]  {air.segment_end}
      }
    }
    return
  }
}
