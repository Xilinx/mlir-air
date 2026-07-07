//===- air_split_l2_memref_launch_offset.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="max-launch-channels-mm2s=16 max-launch-channels-s2mm=16" | FileCheck %s

// A single air.launch with iteration count > 1 (grid > 1) carries a per-
// iteration base offset on its L3 channel puts/gets: launch_idx * tile,
// expressed as a plain arith SSA value (not an affine.apply). When the pass
// splits the L2 memref across the 8-wide herd, each per-core L3 access offset
// must be base + core * sub_tile -- the launch base must NOT be dropped.
//
// Regression: previously the split overwrote the offset with core * sub_tile
// only (a static 0, 128, ... 896), because a default/contiguous access pattern
// was wrongly assumed to imply a zero base offset. Every launch iteration then
// collapsed onto offset 0, so only the first program's data was moved.

// The per-core sub-tile strides are concrete affine maps base + core*128.
// CHECK-DAG: #[[$ADD128:.+]] = affine_map<()[s0] -> (s0 + 128)>
// CHECK-DAG: #[[$ADD896:.+]] = affine_map<()[s0] -> (s0 + 896)>

// CHECK-LABEL: func.func @vecadd
// The launch base (launch_idx * 1024) must still feed the split offsets.
// CHECK: %[[MUL:.*]] = arith.muli %{{.*}}, %c1024{{.*}} : index
// CHECK: %[[I32:.*]] = arith.index_cast %[[MUL]] : index to i32
// CHECK: %[[BASE:.*]] = arith.index_cast %[[I32]] : i32 to index

// MM2S: core 0 accesses L3 at the launch base directly; the wrap stays at the
// sub-tile size (128), and each subsequent core adds its stride to that base.
// CHECK: air.channel.put {{.*}}[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%[[BASE]]] [%c128{{.*}}] [
// CHECK: %[[P1:.*]] = affine.apply #[[$ADD128]]()[%[[BASE]]]
// CHECK: air.channel.put {{.*}}[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%[[P1]]] [%c128{{.*}}] [
// CHECK: %[[P7:.*]] = affine.apply #[[$ADD896]]()[%[[BASE]]]
// CHECK: air.channel.put {{.*}}[%c7{{.*}}, %c0{{.*}}] (%{{.*}}[%[[P7]]] [%c128{{.*}}] [

// S2MM: the get path threads the same launch base through its split offsets.
// CHECK: air.channel.get {{.*}}[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%[[BASE]]] [%c128{{.*}}] [
// CHECK: %[[G1:.*]] = affine.apply #[[$ADD128]]()[%[[BASE]]]
// CHECK: air.channel.get {{.*}}[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%[[G1]]] [%c128{{.*}}] [

#map = affine_map<()[s0] -> (s0 * 128)>
module {
  air.channel @channel_0 []
  air.channel @channel_1 []
  air.channel @channel_2 [8, 1]
  air.channel @channel_3 [8, 1]
  air.channel @channel_4 [8, 1]
  air.channel @channel_5 []
  func.func @vecadd(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg9, %arg10, %arg11) in (%arg12=%c2, %arg13=%c1, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {id = 1 : i32} {
      %c1024 = arith.constant 1024 : index
      %c1_0 = arith.constant 1 : index
      %1 = arith.index_cast %arg9 : index to i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.muli %2, %c1024 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = air.channel.put async  @channel_0[] (%arg15[%5] [%c1024] [%c1_0]) {id = 1 : i32} : (memref<*xbf16>)
      %7 = air.channel.put async  @channel_1[] (%arg16[%5] [%c1024] [%c1_0]) {id = 2 : i32} : (memref<*xbf16>)
      %8 = air.channel.get async  @channel_5[] (%arg17[%5] [%c1024] [%c1_0]) {id = 3 : i32} : (memref<*xbf16>)
      %9 = air.segment @vecadd_0 async  attributes {id = 2 : i32} {
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<1024xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xbf16, 1 : i32>
        }
        %10 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 4 : i32} : (memref<1024xbf16, 1 : i32>)
        %async_token_2, %results_3 = air.execute -> (memref<1024xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1024xbf16, 1 : i32>
        }
        %11 = air.channel.get async [%async_token_2]  @channel_1[] (%results_3[] [] []) {id = 5 : i32} : (memref<1024xbf16, 1 : i32>)
        %async_token_4, %results_5 = air.execute -> (memref<1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %alloc : memref<1024xbf16, 1>
        }
        %12 = air.wait_all async [%10, %async_token_4]
        %13 = scf.parallel (%arg18) = (%c0) to (%c8) step (%c1_1) init (%12) -> !air.async.token {
          %19 = affine.apply #map()[%arg18]
          %20 = air.channel.put async [%10, %async_token_4]  @channel_2[%arg18, %c0] (%results[%19] [%c128] [%c1_1]) {id = 6 : i32} : (memref<1024xbf16, 1 : i32>)
          scf.reduce(%20 : !air.async.token) {
          ^bb0(%arg19: !air.async.token, %arg20: !air.async.token):
            %21 = air.wait_all async [%arg19, %arg20]
            scf.reduce.return %21 : !air.async.token
          }
        }
        %14 = air.wait_all async [%11, %async_token_4]
        %15 = scf.parallel (%arg18) = (%c0) to (%c8) step (%c1_1) init (%14) -> !air.async.token {
          %19 = affine.apply #map()[%arg18]
          %20 = air.channel.put async [%11, %async_token_4]  @channel_3[%arg18, %c0] (%results_3[%19] [%c128] [%c1_1]) {id = 7 : i32} : (memref<1024xbf16, 1 : i32>)
          scf.reduce(%20 : !air.async.token) {
          ^bb0(%arg19: !air.async.token, %arg20: !air.async.token):
            %21 = air.wait_all async [%arg19, %arg20]
            scf.reduce.return %21 : !air.async.token
          }
        }
        %16 = scf.parallel (%arg18) = (%c0) to (%c8) step (%c1_1) init (%async_token_4) -> !air.async.token {
          %19 = affine.apply #map()[%arg18]
          %20 = air.channel.get async [%async_token_4]  @channel_4[%arg18, %c0] (%results_5[%19] [%c128] [%c1_1]) {id = 8 : i32} : (memref<1024xbf16, 1>)
          scf.reduce(%20 : !air.async.token) {
          ^bb0(%arg19: !air.async.token, %arg20: !air.async.token):
            %21 = air.wait_all async [%arg19, %arg20]
            scf.reduce.return %21 : !air.async.token
          }
        }
        %17 = air.herd @herd_0 async [%async_token_4]  tile (%arg18, %arg19) in (%arg20=%c8, %arg21=%c1_1) attributes {id = 3 : i32} {
          %19 = ub.poison : bf16
          %c0_7 = arith.constant 0 : index
          %c128_8 = arith.constant 128 : index
          %c32 = arith.constant 32 : index
          %async_token_9, %results_10 = air.execute -> (memref<128xbf16, 2>) {
            %alloc = memref.alloc() : memref<128xbf16, 2>
            air.execute_terminator %alloc : memref<128xbf16, 2>
          }
          %20 = air.channel.get async [%async_token_9]  @channel_2[%arg18, %arg19] (%results_10[] [] []) {id = 9 : i32} : (memref<128xbf16, 2>)
          %async_token_11, %results_12 = air.execute -> (memref<128xbf16, 2>) {
            %alloc = memref.alloc() : memref<128xbf16, 2>
            air.execute_terminator %alloc : memref<128xbf16, 2>
          }
          %21 = air.channel.get async [%async_token_11]  @channel_3[%arg18, %arg19] (%results_12[] [] []) {id = 10 : i32} : (memref<128xbf16, 2>)
          %async_token_13, %results_14 = air.execute -> (memref<128xbf16, 2>) {
            %alloc = memref.alloc() : memref<128xbf16, 2>
            air.execute_terminator %alloc : memref<128xbf16, 2>
          }
          %22 = air.wait_all async [%20, %21, %async_token_13]
          %23 = scf.for %arg22 = %c0_7 to %c128_8 step %c32 iter_args(%arg23 = %22) -> (!air.async.token) {
            %subview = memref.subview %results_10[%arg22] [32] [1] : memref<128xbf16, 2> to memref<32xbf16, strided<[1], offset: ?>, 2>
            %subview_18 = memref.subview %results_12[%arg22] [32] [1] : memref<128xbf16, 2> to memref<32xbf16, strided<[1], offset: ?>, 2>
            %subview_19 = memref.subview %results_14[%arg22] [32] [1] : memref<128xbf16, 2> to memref<32xbf16, strided<[1], offset: ?>, 2>
            %async_token_20, %results_21 = air.execute [%arg23] -> (vector<32xbf16>) {
              %27 = vector.transfer_read %subview[%c0_7], %19 {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %27 : vector<32xbf16>
            }
            %async_token_22, %results_23 = air.execute [%arg23] -> (vector<32xbf16>) {
              %27 = vector.transfer_read %subview_18[%c0_7], %19 {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %27 : vector<32xbf16>
            }
            %25 = arith.addf %results_21, %results_23 : vector<32xbf16>
            %async_token_24 = air.execute [%arg23] {
              vector.transfer_write %25, %subview_19[%c0_7] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2>
            }
            %26 = air.wait_all async [%async_token_20, %async_token_22, %async_token_24]
            scf.yield %26 : !air.async.token
          }
          %24 = air.channel.put async [%async_token_13]  @channel_4[%arg18, %arg19] (%results_14[] [] []) {id = 11 : i32} : (memref<128xbf16, 2>)
          %async_token_15 = air.execute [%20] {
            memref.dealloc %results_10 : memref<128xbf16, 2>
          }
          %async_token_16 = air.execute [%21] {
            memref.dealloc %results_12 : memref<128xbf16, 2>
          }
          %async_token_17 = air.execute [%24] {
            memref.dealloc %results_14 : memref<128xbf16, 2>
          }
        }
        %18 = air.channel.put async [%17]  @channel_5[] (%results_5[] [] []) {id = 12 : i32} : (memref<1024xbf16, 1>)
        %async_token_6 = air.execute [%18] {
          memref.dealloc %results_5 : memref<1024xbf16, 1>
        }
      }
    }
    return
  }
}
