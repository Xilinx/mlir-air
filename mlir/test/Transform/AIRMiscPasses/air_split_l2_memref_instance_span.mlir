//===- air_split_l2_memref_instance_span.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" --split-input-file | FileCheck %s

// A channel instance that takes several slices of the buffer one after the
// other owns one part: from the lowest offset it reads to the highest. Here
// core r takes rows [64r, 64r+32) and then [64r+32, 64r+64), so the buffer
// splits in two, and each put's constant offset is rebased onto its part.

// CHECK-LABEL: func.func @two_slices
// CHECK-COUNT-2: memref.alloc() : memref<64x64xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<{{.*}}, 1 : i32>
// CHECK: air.channel.put {{.*}} @chan_a[%{{.*}}, %{{.*}}] (%[[P0:.*]][0, %{{.*}}] [32, 64] [64, 1])
// CHECK: air.channel.put {{.*}} @chan_a[%{{.*}}, %{{.*}}] (%[[P0]][32, %{{.*}}] [32, 64] [64, 1])
// CHECK: air.channel.put {{.*}} @chan_a[%{{.*}}, %{{.*}}] (%[[P1:.*]][0, %{{.*}}] [32, 64] [64, 1])
// CHECK: air.channel.put {{.*}} @chan_a[%{{.*}}, %{{.*}}] (%[[P1]][32, %{{.*}}] [32, 64] [64, 1])

module {
  air.channel @chan_l3 []
  air.channel @chan_a [2, 1]
  func.func @two_slices(%arg0: memref<128x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0) : memref<128x64xbf16> {
      %1 = air.channel.put async @chan_l3[] (%arg7[] [] []) : (memref<128x64xbf16>)
      %2 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<128x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<128x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<128x64xbf16, 1 : i32>
        }
        %3 = air.channel.get async [%async_token] @chan_l3[] (%results[] [] []) : (memref<128x64xbf16, 1 : i32>)
        %4 = air.channel.put async [%3] @chan_a[%c0, %c0] (%results[%c0, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %c32 = arith.constant 32 : index
        %5 = air.channel.put async [%4] @chan_a[%c0, %c0] (%results[%c32, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %c64 = arith.constant 64 : index
        %6 = air.channel.put async [%3] @chan_a[%c1_0, %c0] (%results[%c64, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %c96 = arith.constant 96 : index
        %7 = air.channel.put async [%6] @chan_a[%c1_0, %c0] (%results[%c96, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %8 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c1_0) {
          %async_token_1, %results_2 = air.execute -> (memref<32x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<32x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<32x64xbf16, 2 : i32>
          }
          %9 = air.channel.get async [%async_token_1] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %10 = air.channel.get async [%9] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %async_token_3 = air.execute [%10] {
            memref.dealloc %results_2 : memref<32x64xbf16, 2 : i32>
          }
        }
        %async_token_4 = air.execute [%5, %7, %8] {
          memref.dealloc %results : memref<128x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// The same parts, walked by a loop whose offset is `k floordiv 4` over a
// step-4 loop: each trip advances one slice. The loop must keep its bounds
// (or have its map rewritten with them); normalizing it to step 1 under the
// unchanged map would read row 0 on every trip.

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * 32)>
// CHECK-LABEL: func.func @floordiv_loop
// CHECK-COUNT-2: memref.alloc() : memref<64x64xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<{{.*}}, 1 : i32>
// CHECK: scf.for %[[K0:.*]] = %{{.*}} to %c8{{.*}} step %c4
// CHECK: %[[R0:.*]] = affine.apply #[[MAP]]()[%[[K0]]]
// CHECK: air.channel.put {{.*}} @chan_a[%{{.*}}, %{{.*}}] (%{{.*}}[%[[R0]], %{{.*}}] [32, 64] [64, 1])
// CHECK: scf.for %[[K1:.*]] = %{{.*}} to %c8{{.*}} step %c4
// CHECK: %[[R1:.*]] = affine.apply #[[MAP]]()[%[[K1]]]
// CHECK: air.channel.put {{.*}} @chan_a[%{{.*}}, %{{.*}}] (%{{.*}}[%[[R1]], %{{.*}}] [32, 64] [64, 1])

#map0 = affine_map<()[s0] -> ((s0 floordiv 4) * 32)>
#map1 = affine_map<()[s0] -> ((s0 floordiv 4) * 32 + 64)>
module {
  air.channel @chan_l3 []
  air.channel @chan_a [2, 1]
  func.func @floordiv_loop(%arg0: memref<128x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0) : memref<128x64xbf16> {
      %1 = air.channel.put async @chan_l3[] (%arg7[] [] []) : (memref<128x64xbf16>)
      %2 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<128x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<128x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<128x64xbf16, 1 : i32>
        }
        %3 = air.channel.get async [%async_token] @chan_l3[] (%results[] [] []) : (memref<128x64xbf16, 1 : i32>)
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %4 = scf.for %k = %c0 to %c8 step %c4 iter_args(%t = %3) -> (!air.async.token) {
          %r = affine.apply #map0()[%k]
          %p = air.channel.put async [%t] @chan_a[%c0, %c0] (%results[%r, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
          scf.yield %p : !air.async.token
        }
        %5 = scf.for %k = %c0 to %c8 step %c4 iter_args(%t = %3) -> (!air.async.token) {
          %r = affine.apply #map1()[%k]
          %p = air.channel.put async [%t] @chan_a[%c1_0, %c0] (%results[%r, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
          scf.yield %p : !air.async.token
        }
        %8 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c1_0) {
          %async_token_1, %results_2 = air.execute -> (memref<32x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<32x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<32x64xbf16, 2 : i32>
          }
          %9 = air.channel.get async [%async_token_1] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %10 = air.channel.get async [%9] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %async_token_3 = air.execute [%10] {
            memref.dealloc %results_2 : memref<32x64xbf16, 2 : i32>
          }
        }
        %async_token_4 = air.execute [%4, %5, %8] {
          memref.dealloc %results : memref<128x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Parts as big as the buffer over the instances, 64 rows here, would start
// at rows 32 and 96, and the second would run past the buffer's end: each
// slice is its own part instead.

// CHECK-LABEL: func.func @near_end
// CHECK-COUNT-2: memref.alloc() : memref<32x64xbf16, 1 : i32>
// CHECK-NOT: memref.alloc() : memref<{{.*}}, 1 : i32>

module {
  air.channel @chan_l3 []
  air.channel @chan_a [2, 1]
  func.func @near_end(%arg0: memref<128x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0) : memref<128x64xbf16> {
      %1 = air.channel.put async @chan_l3[] (%arg7[] [] []) : (memref<128x64xbf16>)
      %2 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<128x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<128x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<128x64xbf16, 1 : i32>
        }
        %3 = air.channel.get async [%async_token] @chan_l3[] (%results[] [] []) : (memref<128x64xbf16, 1 : i32>)
        %c32 = arith.constant 32 : index
        %c96 = arith.constant 96 : index
        %4 = air.channel.put async [%3] @chan_a[%c0, %c0] (%results[%c32, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %5 = air.channel.put async [%3] @chan_a[%c1_0, %c0] (%results[%c96, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %8 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c1_0) {
          %async_token_1, %results_2 = air.execute -> (memref<32x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<32x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<32x64xbf16, 2 : i32>
          }
          %9 = air.channel.get async [%async_token_1] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %async_token_3 = air.execute [%9] {
            memref.dealloc %results_2 : memref<32x64xbf16, 2 : i32>
          }
        }
        %async_token_4 = air.execute [%4, %5, %8] {
          memref.dealloc %results : memref<128x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// The instances' transfers in reverse order: the parts still follow the
// buffer's order, part i taking L3's rows from 64 i (later passes, padding
// the last block, place a part by its index).

// CHECK-LABEL: func.func @reversed
// CHECK: air.channel.put {{.*}}@[[CH:[a-z_0-9]+]][%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%{{.*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c64
// CHECK: air.channel.put {{.*}}@[[CH]][%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] (%{{.*}}[%c64{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c64
// CHECK: air.channel.get {{.*}}@[[CH]][%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%[[P0:[a-z_0-9]+]][
// CHECK: air.channel.get {{.*}}@[[CH]][%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] (%[[P1:[a-z_0-9]+]][
// CHECK: @chan_a[%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] (%[[P1]][0,
// CHECK: @chan_a[%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] (%[[P1]][32,
// CHECK: @chan_a[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%[[P0]][0,
// CHECK: @chan_a[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%[[P0]][32,

module {
  air.channel @chan_l3 []
  air.channel @chan_a [2, 1]
  func.func @reversed(%arg0: memref<128x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0) : memref<128x64xbf16> {
      %1 = air.channel.put async @chan_l3[] (%arg7[] [] []) : (memref<128x64xbf16>)
      %2 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<128x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<128x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<128x64xbf16, 1 : i32>
        }
        %3 = air.channel.get async [%async_token] @chan_l3[] (%results[] [] []) : (memref<128x64xbf16, 1 : i32>)
        %c64 = arith.constant 64 : index
        %c96 = arith.constant 96 : index
        %c32 = arith.constant 32 : index
        %4 = air.channel.put async [%3] @chan_a[%c1_0, %c0] (%results[%c64, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %5 = air.channel.put async [%4] @chan_a[%c1_0, %c0] (%results[%c96, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %6 = air.channel.put async [%3] @chan_a[%c0, %c0] (%results[%c0, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %7 = air.channel.put async [%6] @chan_a[%c0, %c0] (%results[%c32, %c0] [32, 64] [64, 1]) : (memref<128x64xbf16, 1 : i32>)
        %8 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c2, %sy=%c1_0) {
          %async_token_1, %results_2 = air.execute -> (memref<32x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<32x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<32x64xbf16, 2 : i32>
          }
          %9 = air.channel.get async [%async_token_1] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %10 = air.channel.get async [%9] @chan_a[%tx, %ty] (%results_2[] [] []) : (memref<32x64xbf16, 2 : i32>)
          %async_token_3 = air.execute [%10] {
            memref.dealloc %results_2 : memref<32x64xbf16, 2 : i32>
          }
        }
        %async_token_4 = air.execute [%5, %7, %8] {
          memref.dealloc %results : memref<128x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
