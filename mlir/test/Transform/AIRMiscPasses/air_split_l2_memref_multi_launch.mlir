//===- air_split_l2_memref_multi_launch.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" | FileCheck %s

// Test that air-split-l2-memref correctly handles multiple launches with
// independent segments. Each segment's L2 memref should be split based on
// the channels within that segment.

// CHECK-LABEL: func.func @attention
// First launch should have segment @npu_mm_exact_0 with split L2 memrefs
// CHECK: air.launch async {{.*}} attributes {id = 1 : i32}
// CHECK: air.segment @npu_mm_exact_0
// The original 256x256 memrefs should be split into 4 64x256 and 4 256x64 memrefs
// CHECK-COUNT-4: memref.alloc() : memref<64x256xbf16, 1 : i32>
// CHECK-COUNT-4: memref.alloc() : memref<256x64xbf16, 1 : i32>
// CHECK-COUNT-4: memref.alloc() : memref<64x256xbf16, 1>
// CHECK: air.herd @herd_0

// Second launch should have segment @softmax with split L2 memrefs
// CHECK: air.launch async {{.*}} attributes {id = 6 : i32}
// CHECK: air.segment @softmax
// The original 4x256 memrefs should be split into 4 1x256 memrefs
// CHECK-COUNT-4: memref.alloc() : memref<1x256xbf16, 1>
// CHECK-COUNT-4: memref.alloc() : memref<1x256xbf16, 1>
// CHECK: air.herd @softmax_herd

#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
#map3 = affine_map<()[s0, s1] -> (s0 * 8 + s1 + 1)>
#map4 = affine_map<(d0) -> (d0 * 64)>
#map5 = affine_map<(d0) -> (d0 * 512)>
#map6 = affine_map<()[s0] -> (s0 * 512 + 512)>
#map7 = affine_map<()[s0] -> (s0 * 64 + 64)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map11 = affine_map<()[s0] -> (s0 * 8)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set7 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 3 == 0)>
module {
  air.channel @channel_0 []
  air.channel @channel_1 []
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_4 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_5 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_6 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_8 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_9 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_10 [4, 4]
  air.channel @channel_11 []
  air.channel @channel_12 []
  air.channel @channel_13 [4, 1]
  air.channel @channel_14 [4, 1]
  air.channel @channel_15 []
  func.func @attention(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xbf16>, %arg3: memref<256x256xbf16>) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = air.launch async (%arg4, %arg5, %arg6) in (%arg7=%c1, %arg8=%c1, %arg9=%c1) args(%arg10=%arg0, %arg11=%arg1, %arg12=%arg2) : memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 1 : i32} {
      %c1_0 = arith.constant 1 : index
      %c65536 = arith.constant 65536 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c64_1 = arith.constant 64 : index
      %2 = arith.muli %arg4, %c65536 : index
      %3 = air.wait_all async 
      %4 = scf.for %arg13 = %c0 to %c256 step %c64_1 iter_args(%arg14 = %3) -> (!air.async.token) {
        %10 = affine.apply #map()[%2, %arg13]
        %11 = air.channel.put async [%arg14]  @channel_0[] (%arg10[%c0, %10] [%c256, %c64_1] [%c256, %c1_0]) {id = 1 : i32} : (memref<256x256xbf16>)
        scf.yield %11 : !air.async.token
      }
      %5 = arith.muli %arg5, %c256 : index
      %6 = air.wait_all async 
      %7 = scf.for %arg13 = %c0 to %c256 step %c64_1 iter_args(%arg14 = %6) -> (!air.async.token) {
        %10 = air.channel.put async [%arg14]  @channel_1[] (%arg11[%arg13, %5] [%c64_1, %c256] [%c256, %c1_0]) {id = 2 : i32} : (memref<256x256xbf16>)
        scf.yield %10 : !air.async.token
      }
      %8 = air.channel.get async  @channel_11[] (%arg12[] [] []) {id = 3 : i32} : (memref<256x256xbf16>)
      %9 = air.segment @npu_mm_exact_0 async  attributes {id = 2 : i32} {
        %c192 = arith.constant 192 : index
        %c128 = arith.constant 128 : index
        %c2048 = arith.constant 2048 : index
        %c8 = arith.constant 8 : index
        %c1_2 = arith.constant 1 : index
        %c256_3 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c0_4 = arith.constant 0 : index
        %c64_5 = arith.constant 64 : index
        %async_token, %results = air.execute -> (memref<256x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x256xbf16, 1 : i32>
        }
        %async_token_6, %results_7 = air.execute -> (memref<256x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x256xbf16, 1 : i32>
        }
        %async_token_8, %results_9 = air.execute -> (memref<256x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<256x256xbf16, 1>
          air.execute_terminator %alloc : memref<256x256xbf16, 1>
        }
        %async_token_10, %results_11 = air.execute -> (memref<32x32x8x8xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32x8x8xbf16, 2>
          air.execute_terminator %alloc : memref<32x32x8x8xbf16, 2>
        }
        %10 = air.wait_all async [%async_token, %async_token_6] 
        %11 = scf.for %arg13 = %c0_4 to %c256_3 step %c64_5 iter_args(%arg14 = %10) -> (!air.async.token) {
          %16 = air.channel.get async [%arg14]  @channel_0[] (%results[%c0_4, %arg13] [%c256_3, %c64_5] [%c256_3, %c1_2]) {id = 4 : i32} : (memref<256x256xbf16, 1 : i32>)
          %17 = air.channel.put async [%16]  @channel_2[] (%results[%c0_4, %c0_4, %c0_4, %arg13] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set, id = 6 : i32} : (memref<256x256xbf16, 1 : i32>)
          %18 = air.channel.put async [%16]  @channel_3[] (%results[%c0_4, %c0_4, %c64_5, %arg13] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set1, id = 7 : i32} : (memref<256x256xbf16, 1 : i32>)
          %19 = air.channel.put async [%16]  @channel_4[] (%results[%c0_4, %c0_4, %c128, %arg13] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set2, id = 8 : i32} : (memref<256x256xbf16, 1 : i32>)
          %20 = air.channel.put async [%16]  @channel_5[] (%results[%c0_4, %c0_4, %c192, %arg13] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set3, id = 9 : i32} : (memref<256x256xbf16, 1 : i32>)
          %21 = air.wait_all async [%17, %18, %19, %20] 
          scf.yield %21 : !air.async.token
        }
        %12 = scf.for %arg13 = %c0_4 to %c256_3 step %c64_5 iter_args(%arg14 = %10) -> (!air.async.token) {
          %16 = air.channel.get async [%arg14]  @channel_1[] (%results_7[%arg13, %c0_4] [%c64_5, %c256_3] [%c256_3, %c1_2]) {id = 5 : i32} : (memref<256x256xbf16, 1 : i32>)
          %17 = air.channel.put async [%16]  @channel_6[] (%results_7[%c0_4, %c0_4, %arg13, %c0_4] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set4, id = 10 : i32} : (memref<256x256xbf16, 1 : i32>)
          %18 = air.channel.put async [%16]  @channel_7[] (%results_7[%c0_4, %c0_4, %arg13, %c64_5] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set5, id = 11 : i32} : (memref<256x256xbf16, 1 : i32>)
          %19 = air.channel.put async [%16]  @channel_8[] (%results_7[%c0_4, %c0_4, %arg13, %c128] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set6, id = 12 : i32} : (memref<256x256xbf16, 1 : i32>)
          %20 = air.channel.put async [%16]  @channel_9[] (%results_7[%c0_4, %c0_4, %arg13, %c192] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_3, %c1_2]) {broadcast_set = #set7, id = 13 : i32} : (memref<256x256xbf16, 1 : i32>)
          %21 = air.wait_all async [%17, %18, %19, %20] 
          scf.yield %21 : !air.async.token
        }
        %13 = scf.parallel (%arg13, %arg14) = (%c0_4, %c0_4) to (%c4, %c4) step (%c1_2, %c1_2) init (%async_token_8) -> !air.async.token {
          %16 = affine.apply #map1()[%arg13]
          %17 = affine.apply #map1()[%arg14]
          %18 = air.channel.get async [%async_token_8]  @channel_10[%arg13, %arg14] (%results_9[%16, %17] [%c64_5, %c64_5] [%c256_3, %c1_2]) {id = 22 : i32} : (memref<256x256xbf16, 1>)
          scf.reduce(%18 : !air.async.token) {
          ^bb0(%arg15: !air.async.token, %arg16: !air.async.token):
            %19 = air.wait_all async [%arg15, %arg16] 
            scf.reduce.return %19 : !air.async.token
          }
        }
        %14 = air.herd @herd_0 async [%async_token_10]  tile (%arg13, %arg14) in (%arg15=%c4, %arg16=%c4) args(%arg17=%results_11) : memref<32x32x8x8xbf16, 2> attributes {id = 5 : i32} {
          %c2048_14 = arith.constant 2048 : index
          %c256_15 = arith.constant 256 : index
          %c64_16 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %16 = ub.poison : bf16
          %c2 = arith.constant 2 : index
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xbf16>
          %c0_17 = arith.constant 0 : index
          %c8_18 = arith.constant 8 : index
          %c1_19 = arith.constant 1 : index
          %17 = air.wait_all async 
          %18 = scf.for %arg18 = %c0_17 to %c8_18 step %c1_19 iter_args(%arg19 = %17) -> (!air.async.token) {
            %24 = scf.for %arg20 = %c0_17 to %c8_18 step %c1_19 iter_args(%arg21 = %arg19) -> (!air.async.token) {
              %25 = affine.apply #map2()[%arg20, %arg14]
              %26 = affine.apply #map2()[%arg18, %arg13]
              %async_token_20 = air.execute [%arg21] {
                vector.transfer_write %cst, %arg17[%25, %26, %c0_17, %c0_17] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
              }
              scf.yield %async_token_20 : !air.async.token
            }
            scf.yield %24 : !air.async.token
          }
          %19 = air.wait_all async 
          %20 = scf.for %arg18 = %c0_17 to %c256_15 step %c64_16 iter_args(%arg19 = %19) -> (!air.async.token) {
            %async_token_20, %results_21 = air.execute [%arg19] -> (memref<8x8x8x8xbf16, 2>) {
              %alloc = memref.alloc() : memref<8x8x8x8xbf16, 2>
              air.execute_terminator %alloc : memref<8x8x8x8xbf16, 2>
            }
            %24 = affine.if #set()[%arg13, %arg14] -> !air.async.token {
              %29 = air.channel.get async [%async_token_20]  @channel_2[%arg13, %arg14] (%results_21[] [] []) {id = 14 : i32} : (memref<8x8x8x8xbf16, 2>)
              affine.yield %29 : !air.async.token
            } else {
              %29 = affine.if #set1()[%arg13, %arg14] -> !air.async.token {
                %30 = air.channel.get async [%async_token_20]  @channel_3[%arg13, %arg14] (%results_21[] [] []) {id = 15 : i32} : (memref<8x8x8x8xbf16, 2>)
                affine.yield %30 : !air.async.token
              } else {
                %30 = affine.if #set2()[%arg13, %arg14] -> !air.async.token {
                  %31 = air.channel.get async [%async_token_20]  @channel_4[%arg13, %arg14] (%results_21[] [] []) {id = 16 : i32} : (memref<8x8x8x8xbf16, 2>)
                  affine.yield %31 : !air.async.token
                } else {
                  %31 = air.channel.get async [%async_token_20]  @channel_5[%arg13, %arg14] (%results_21[] [] []) {id = 17 : i32} : (memref<8x8x8x8xbf16, 2>)
                  affine.yield %31 : !air.async.token
                }
                affine.yield %30 : !air.async.token
              }
              affine.yield %29 : !air.async.token
            }
            %async_token_22, %results_23 = air.execute [%arg19] -> (memref<8x8x8x8xbf16, 2>) {
              %alloc = memref.alloc() : memref<8x8x8x8xbf16, 2>
              air.execute_terminator %alloc : memref<8x8x8x8xbf16, 2>
            }
            %25 = affine.if #set4()[%arg13, %arg14] -> !air.async.token {
              %29 = air.channel.get async [%async_token_22]  @channel_6[%arg13, %arg14] (%results_23[] [] []) {id = 18 : i32} : (memref<8x8x8x8xbf16, 2>)
              affine.yield %29 : !air.async.token
            } else {
              %29 = affine.if #set5()[%arg13, %arg14] -> !air.async.token {
                %30 = air.channel.get async [%async_token_22]  @channel_7[%arg13, %arg14] (%results_23[] [] []) {id = 19 : i32} : (memref<8x8x8x8xbf16, 2>)
                affine.yield %30 : !air.async.token
              } else {
                %30 = affine.if #set6()[%arg13, %arg14] -> !air.async.token {
                  %31 = air.channel.get async [%async_token_22]  @channel_8[%arg13, %arg14] (%results_23[] [] []) {id = 20 : i32} : (memref<8x8x8x8xbf16, 2>)
                  affine.yield %31 : !air.async.token
                } else {
                  %31 = air.channel.get async [%async_token_22]  @channel_9[%arg13, %arg14] (%results_23[] [] []) {id = 21 : i32} : (memref<8x8x8x8xbf16, 2>)
                  affine.yield %31 : !air.async.token
                }
                affine.yield %30 : !air.async.token
              }
              affine.yield %29 : !air.async.token
            }
            %26 = air.wait_all async [%24, %25] 
            %27 = scf.for %arg20 = %c0_17 to %c8_18 step %c2 iter_args(%arg21 = %26) -> (!air.async.token) {
              %29 = scf.for %arg22 = %c0_17 to %c8_18 step %c2 iter_args(%arg23 = %arg21) -> (!air.async.token) {
                %30 = affine.apply #map2()[%arg22, %arg14]
                %31 = affine.apply #map2()[%arg20, %arg13]
                %async_token_26, %results_27 = air.execute [%arg23] -> (vector<1x1x8x8xbf16>) {
                  %56 = vector.transfer_read %arg17[%30, %31, %c0_17, %c0_17], %16 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                  air.execute_terminator %56 : vector<1x1x8x8xbf16>
                }
                %32 = affine.apply #map3()[%arg14, %arg22]
                %async_token_28, %results_29 = air.execute [%async_token_26] -> (vector<1x1x8x8xbf16>) {
                  %56 = vector.transfer_read %arg17[%32, %31, %c0_17, %c0_17], %16 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                  air.execute_terminator %56 : vector<1x1x8x8xbf16>
                }
                %33 = affine.apply #map3()[%arg13, %arg20]
                %async_token_30, %results_31 = air.execute [%async_token_28] -> (vector<1x1x8x8xbf16>) {
                  %56 = vector.transfer_read %arg17[%30, %33, %c0_17, %c0_17], %16 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                  air.execute_terminator %56 : vector<1x1x8x8xbf16>
                }
                %async_token_32, %results_33 = air.execute [%async_token_30] -> (vector<1x1x8x8xbf16>) {
                  %56 = vector.transfer_read %arg17[%32, %33, %c0_17, %c0_17], %16 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                  air.execute_terminator %56 : vector<1x1x8x8xbf16>
                }
                %34 = arith.extf %results_27 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %35 = arith.extf %results_29 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %36 = arith.extf %results_31 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %37 = arith.extf %results_33 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %38 = vector.shape_cast %34 : vector<1x1x8x8xf32> to vector<64xf32>
                %39 = vector.shape_cast %35 : vector<1x1x8x8xf32> to vector<64xf32>
                %40 = vector.shape_cast %36 : vector<1x1x8x8xf32> to vector<64xf32>
                %41 = vector.shape_cast %37 : vector<1x1x8x8xf32> to vector<64xf32>
                %collapse_shape = memref.collapse_shape %results_21 [[0, 1, 2, 3]] : memref<8x8x8x8xbf16, 2> into memref<4096xbf16, 2>
                %42 = affine.apply #map4(%arg20)
                %collapse_shape_34 = memref.collapse_shape %results_23 [[0, 1, 2, 3]] : memref<8x8x8x8xbf16, 2> into memref<4096xbf16, 2>
                %43 = affine.apply #map5(%arg22)
                %44 = affine.apply #map6()[%arg22]
                %45 = affine.apply #map7()[%arg20]
                %46:9 = scf.for %arg24 = %c0_17 to %c8_18 step %c1_19 iter_args(%arg25 = %38, %arg26 = %39, %arg27 = %40, %arg28 = %41, %arg29 = %42, %arg30 = %43, %arg31 = %44, %arg32 = %45, %arg33 = %arg23) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, index, index, index, index, !air.async.token) {
                  %56 = vector.shape_cast %arg25 : vector<64xf32> to vector<1x1x8x8xf32>
                  %57 = vector.shape_cast %arg26 : vector<64xf32> to vector<1x1x8x8xf32>
                  %58 = vector.shape_cast %arg27 : vector<64xf32> to vector<1x1x8x8xf32>
                  %59 = vector.shape_cast %arg28 : vector<64xf32> to vector<1x1x8x8xf32>
                  %async_token_39, %results_40 = air.execute [%arg33] -> (vector<64xbf16>) {
                    %77 = vector.transfer_read %collapse_shape[%arg29], %16 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                    air.execute_terminator %77 : vector<64xbf16>
                  }
                  %60 = vector.shape_cast %results_40 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %61 = arith.addi %arg29, %c512 : index
                  %async_token_41, %results_42 = air.execute [%arg33] -> (vector<64xbf16>) {
                    %77 = vector.transfer_read %collapse_shape_34[%arg30], %16 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                    air.execute_terminator %77 : vector<64xbf16>
                  }
                  %62 = vector.shape_cast %results_42 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %63 = arith.addi %arg30, %c64_16 : index
                  %64 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %60, %62, %56 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %async_token_43, %results_44 = air.execute [%async_token_41] -> (vector<64xbf16>) {
                    %77 = vector.transfer_read %collapse_shape_34[%arg31], %16 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                    air.execute_terminator %77 : vector<64xbf16>
                  }
                  %65 = vector.shape_cast %results_44 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %66 = arith.addi %arg31, %c64_16 : index
                  %67 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %60, %65, %57 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %async_token_45, %results_46 = air.execute [%async_token_39] -> (vector<64xbf16>) {
                    %77 = vector.transfer_read %collapse_shape[%arg32], %16 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                    air.execute_terminator %77 : vector<64xbf16>
                  }
                  %68 = vector.shape_cast %results_46 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %69 = arith.addi %arg32, %c512 : index
                  %70 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %68, %62, %58 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %71 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %68, %65, %59 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %72 = vector.shape_cast %64 : vector<1x1x8x8xf32> to vector<64xf32>
                  %73 = vector.shape_cast %67 : vector<1x1x8x8xf32> to vector<64xf32>
                  %74 = vector.shape_cast %70 : vector<1x1x8x8xf32> to vector<64xf32>
                  %75 = vector.shape_cast %71 : vector<1x1x8x8xf32> to vector<64xf32>
                  %76 = air.wait_all async [%async_token_43, %async_token_45] 
                  scf.yield %72, %73, %74, %75, %61, %63, %66, %69, %76 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, index, index, index, index, !air.async.token
                }
                %47 = vector.shape_cast %46#0 : vector<64xf32> to vector<1x1x8x8xf32>
                %48 = vector.shape_cast %46#1 : vector<64xf32> to vector<1x1x8x8xf32>
                %49 = vector.shape_cast %46#2 : vector<64xf32> to vector<1x1x8x8xf32>
                %50 = vector.shape_cast %46#3 : vector<64xf32> to vector<1x1x8x8xf32>
                %51 = arith.truncf %50 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %52 = arith.truncf %49 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %53 = arith.truncf %48 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %54 = arith.truncf %47 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %async_token_35 = air.execute [%async_token_32] {
                  vector.transfer_write %51, %arg17[%32, %33, %c0_17, %c0_17] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                }
                %async_token_36 = air.execute [%async_token_35] {
                  vector.transfer_write %52, %arg17[%30, %33, %c0_17, %c0_17] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                }
                %async_token_37 = air.execute [%async_token_36] {
                  vector.transfer_write %53, %arg17[%32, %31, %c0_17, %c0_17] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                }
                %async_token_38 = air.execute [%async_token_37] {
                  vector.transfer_write %54, %arg17[%30, %31, %c0_17, %c0_17] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                }
                %55 = air.wait_all async [%46#8, %async_token_38] 
                scf.yield %55 : !air.async.token
              }
              scf.yield %29 : !air.async.token
            }
            %async_token_24 = air.execute [%24] {
              memref.dealloc %results_21 : memref<8x8x8x8xbf16, 2>
            }
            %async_token_25 = air.execute [%25] {
              memref.dealloc %results_23 : memref<8x8x8x8xbf16, 2>
            }
            %28 = air.wait_all async [%27, %async_token_24, %async_token_25] 
            scf.yield %28 : !air.async.token
          }
          %21 = affine.apply #map11()[%arg13]
          %22 = affine.apply #map11()[%arg14]
          %23 = air.channel.put async  @channel_10[%arg13, %arg14] (%arg17[%21, %c0_17, %22, %c0_17] [%c8_18, %c8_18, %c8_18, %c8_18] [%c64_16, %c8_18, %c2048_14, %c1_19]) {id = 23 : i32} : (memref<32x32x8x8xbf16, 2>)
        }
        %15 = air.channel.put async [%13]  @channel_11[] (%results_9[] [] []) {id = 24 : i32} : (memref<256x256xbf16, 1>)
        %async_token_12 = air.execute [%15] {
          memref.dealloc %results_9 : memref<256x256xbf16, 1>
        }
        %async_token_13 = air.execute {
          memref.dealloc %results_11 : memref<32x32x8x8xbf16, 2>
        }
      }
    }
    %1 = air.launch async [%0] (%arg4, %arg5, %arg6) in (%arg7=%c64, %arg8=%c1, %arg9=%c1) args(%arg10=%arg2, %arg11=%arg3) : memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 6 : i32} {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %arg4, %c4 : index
      %3 = air.channel.put async  @channel_12[] (%arg10[%2, %c0] [%c4, %c256] [%c256, %c1_0]) {id = 25 : i32} : (memref<256x256xbf16>)
      %4 = air.channel.get async  @channel_15[] (%arg11[%2, %c0] [%c4, %c256] [%c256, %c1_0]) {id = 26 : i32} : (memref<256x256xbf16>)
      %5 = air.segment @softmax async  attributes {id = 7 : i32} {
        %c0_1 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c4_3 = arith.constant 4 : index
        %c256_4 = arith.constant 256 : index
        %async_token, %results = air.execute -> (memref<4x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<4x256xbf16, 1>
          air.execute_terminator %alloc : memref<4x256xbf16, 1>
        }
        %async_token_5, %results_6 = air.execute -> (memref<4x256xbf16, 1>) {
          %alloc = memref.alloc() : memref<4x256xbf16, 1>
          air.execute_terminator %alloc : memref<4x256xbf16, 1>
        }
        %6 = air.channel.get async [%async_token]  @channel_12[] (%results[] [] []) {id = 27 : i32} : (memref<4x256xbf16, 1>)
        %7 = scf.parallel (%arg12) = (%c0_1) to (%c4_3) step (%c1_2) init (%6) -> !air.async.token {
          %11 = air.channel.put async [%6]  @channel_13[%arg12, %c0_1] (%results[%arg12, %c0_1] [%c1_2, %c256_4] [%c256_4, %c1_2]) {id = 28 : i32} : (memref<4x256xbf16, 1>)
          scf.reduce(%11 : !air.async.token) {
          ^bb0(%arg13: !air.async.token, %arg14: !air.async.token):
            %12 = air.wait_all async [%arg13, %arg14] 
            scf.reduce.return %12 : !air.async.token
          }
        }
        %8 = scf.parallel (%arg12) = (%c0_1) to (%c4_3) step (%c1_2) init (%async_token_5) -> !air.async.token {
          %11 = air.channel.get async [%async_token_5]  @channel_14[%arg12, %c0_1] (%results_6[%arg12, %c0_1] [%c1_2, %c256_4] [%c256_4, %c1_2]) {id = 29 : i32} : (memref<4x256xbf16, 1>)
          scf.reduce(%11 : !air.async.token) {
          ^bb0(%arg13: !air.async.token, %arg14: !air.async.token):
            %12 = air.wait_all async [%arg13, %arg14] 
            scf.reduce.return %12 : !air.async.token
          }
        }
        %9 = air.herd @softmax_herd async [%async_token_5, %6]  tile (%arg12, %arg13) in (%arg14=%c4_3, %arg15=%c1_2) attributes {id = 8 : i32} {
          %c0_9 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c256_10 = arith.constant 256 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_11 = arith.constant 0xFF800000 : f32
          %11 = ub.poison : bf16
          %async_token_12, %results_13 = air.execute -> (memref<1x256xbf16, 2>) {
            %alloc = memref.alloc() : memref<1x256xbf16, 2>
            air.execute_terminator %alloc : memref<1x256xbf16, 2>
          }
          %async_token_14, %results_15 = air.execute -> (memref<1x256xbf16, 2>) {
            %alloc = memref.alloc() : memref<1x256xbf16, 2>
            air.execute_terminator %alloc : memref<1x256xbf16, 2>
          }
          %12 = air.channel.get async [%async_token_12]  @channel_13[%arg12, %arg13] (%results_13[] [] []) {id = 30 : i32} : (memref<1x256xbf16, 2>)
          %async_token_16, %results_17 = air.execute -> (memref<1xf32, 2>) {
            %alloc = memref.alloc() : memref<1xf32, 2>
            air.execute_terminator %alloc : memref<1xf32, 2>
          }
          %async_token_18 = air.execute [%async_token_16] {
            memref.store %cst_11, %results_17[%c0_9] : memref<1xf32, 2>
          }
          %13 = air.wait_all async [%12, %async_token_18] 
          %14 = scf.for %arg16 = %c0_9 to %c256_10 step %c32 iter_args(%arg17 = %13) -> (!air.async.token) {
            %subview = memref.subview %results_13[0, %arg16] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %async_token_26, %results_27 = air.execute [%arg17] -> (vector<32xbf16>) {
              %24 = vector.transfer_read %subview[%c0_9, %c0_9], %11 {in_bounds = [true]} : memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %24 : vector<32xbf16>
            }
            %async_token_28, %results_29 = air.execute [%arg17] -> (f32) {
              %24 = memref.load %results_17[%c0_9] : memref<1xf32, 2>
              air.execute_terminator %24 : f32
            }
            %20 = arith.truncf %results_29 : f32 to bf16
            %21 = vector.reduction <maxnumf>, %results_27, %20 : vector<32xbf16> into bf16
            %22 = arith.extf %21 : bf16 to f32
            %async_token_30 = air.execute [%async_token_28] {
              memref.store %22, %results_17[%c0_9] : memref<1xf32, 2>
            }
            %23 = air.wait_all async [%async_token_26, %async_token_30] 
            scf.yield %23 : !air.async.token
          }
          %async_token_19, %results_20 = air.execute -> (memref<1xf32, 2>) {
            %alloc = memref.alloc() : memref<1xf32, 2>
            air.execute_terminator %alloc : memref<1xf32, 2>
          }
          %async_token_21 = air.execute [%async_token_19] {
            memref.store %cst, %results_20[%c0_9] : memref<1xf32, 2>
          }
          %15 = air.wait_all async [%14, %async_token_21] 
          %16 = scf.for %arg16 = %c0_9 to %c256_10 step %c32 iter_args(%arg17 = %15) -> (!air.async.token) {
            %subview = memref.subview %results_13[0, %arg16] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %async_token_26, %results_27 = air.execute [%arg17] -> (vector<32xbf16>) {
              %29 = vector.transfer_read %subview[%c0_9, %c0_9], %11 {in_bounds = [true]} : memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %29 : vector<32xbf16>
            }
            %async_token_28, %results_29 = air.execute [%arg17] -> (f32) {
              %29 = memref.load %results_17[%c0_9] : memref<1xf32, 2>
              air.execute_terminator %29 : f32
            }
            %async_token_30, %results_31 = air.execute [%arg17] -> (f32) {
              %29 = memref.load %results_20[%c0_9] : memref<1xf32, 2>
              air.execute_terminator %29 : f32
            }
            %20 = arith.extf %results_27 : vector<32xbf16> to vector<32xf32>
            %21 = vector.broadcast %results_29 : f32 to vector<32xf32>
            %22 = arith.subf %20, %21 : vector<32xf32>
            %23 = arith.truncf %22 : vector<32xf32> to vector<32xbf16>
            %24 = math.exp %23 : vector<32xbf16>
            %25 = arith.truncf %results_31 : f32 to bf16
            %26 = vector.reduction <add>, %24, %25 : vector<32xbf16> into bf16
            %27 = arith.extf %26 : bf16 to f32
            %async_token_32 = air.execute [%async_token_30] {
              memref.store %27, %results_20[%c0_9] : memref<1xf32, 2>
            }
            %28 = air.wait_all async [%async_token_26, %async_token_28, %async_token_32] 
            scf.yield %28 : !air.async.token
          }
          %17 = air.wait_all async [%async_token_14, %16] 
          %18 = scf.for %arg16 = %c0_9 to %c256_10 step %c32 iter_args(%arg17 = %17) -> (!air.async.token) {
            %subview = memref.subview %results_13[0, %arg16] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %subview_26 = memref.subview %results_15[0, %arg16] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %async_token_27, %results_28 = air.execute [%arg17] -> (vector<32xbf16>) {
              %30 = vector.transfer_read %subview[%c0_9, %c0_9], %11 {in_bounds = [true]} : memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>, vector<32xbf16>
              air.execute_terminator %30 : vector<32xbf16>
            }
            %async_token_29, %results_30 = air.execute [%arg17] -> (f32) {
              %30 = memref.load %results_17[%c0_9] : memref<1xf32, 2>
              air.execute_terminator %30 : f32
            }
            %async_token_31, %results_32 = air.execute [%arg17] -> (f32) {
              %30 = memref.load %results_20[%c0_9] : memref<1xf32, 2>
              air.execute_terminator %30 : f32
            }
            %20 = arith.extf %results_28 : vector<32xbf16> to vector<32xf32>
            %21 = vector.broadcast %results_30 : f32 to vector<32xf32>
            %22 = arith.subf %20, %21 : vector<32xf32>
            %23 = arith.truncf %22 : vector<32xf32> to vector<32xbf16>
            %24 = math.exp %23 : vector<32xbf16>
            %25 = arith.extf %24 : vector<32xbf16> to vector<32xf32>
            %26 = vector.broadcast %results_32 : f32 to vector<32xf32>
            %27 = arith.divf %25, %26 : vector<32xf32>
            %28 = arith.truncf %27 : vector<32xf32> to vector<32xbf16>
            %async_token_33 = air.execute [%arg17] {
              vector.transfer_write %28, %subview_26[%c0_9, %c0_9] {in_bounds = [true]} : vector<32xbf16>, memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            }
            %29 = air.wait_all async [%async_token_27, %async_token_29, %async_token_31, %async_token_33] 
            scf.yield %29 : !air.async.token
          }
          %19 = air.channel.put async [%async_token_14]  @channel_14[%arg12, %arg13] (%results_15[] [] []) {id = 31 : i32} : (memref<1x256xbf16, 2>)
          %async_token_22 = air.execute [%18] {
            memref.dealloc %results_17 : memref<1xf32, 2>
          }
          %async_token_23 = air.execute [%18] {
            memref.dealloc %results_20 : memref<1xf32, 2>
          }
          %async_token_24 = air.execute [%12] {
            memref.dealloc %results_13 : memref<1x256xbf16, 2>
          }
          %async_token_25 = air.execute [%19] {
            memref.dealloc %results_15 : memref<1x256xbf16, 2>
          }
        }
        %10 = air.channel.put async [%9]  @channel_15[] (%results_6[] [] []) {id = 32 : i32} : (memref<4x256xbf16, 1>)
        %async_token_7 = air.execute [%7] {
          memref.dealloc %results : memref<4x256xbf16, 1>
        }
        %async_token_8 = air.execute [%10] {
          memref.dealloc %results_6 : memref<4x256xbf16, 1>
        }
      }
    }
    return
  }
}
