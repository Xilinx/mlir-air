//===- cascade_broadcast_hoist_constant_slice.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for a crash in air-dma-to-channel when hoisting the external
// half of a broadcast DMA out of a herd whose body nests the channel ops under
// several affine.if/scf.if regions (a cascade GEMV). While collecting the
// constant operands of the backward slice, the pass inserted into the
// `backwardSlice` SetVector from inside the range-for that iterates it; for a
// large enough slice the SetVector's backing storage reallocated mid-iteration,
// invalidating the iterator and causing a SIGSEGV. Collecting the constants
// into a temporary first and inserting afterwards fixes it; the resulting
// channelization is unchanged.

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// The compiler-managed cascade channel is preserved.
// CHECK-DAG: air.channel @chan_cascade [2, 1] {channel_type = "npu_cascade"}
// The broadcast B-vector DMA is split per-cascade-row into broadcast channels.
// CHECK-DAG: air.channel @[[BCAST0:.*]] [1, 1] {broadcast_shape = [2, 1]}
// CHECK-DAG: air.channel @[[BCAST1:.*]] [1, 1] {broadcast_shape = [2, 1]}

// The external (L3) side of the broadcast DMA is hoisted out of the herd, up to
// the launch, as broadcast channel puts.
// CHECK: air.launch
// CHECK: air.channel.put {{.*}}broadcast_set
// CHECK: air.channel.put {{.*}}broadcast_set
// CHECK: air.segment @matvec_cascade_0
// CHECK: air.herd @herd_0
// The cascade put/get survive inside the herd body.
// CHECK: air.channel.put {{.*}}@chan_cascade
// CHECK: air.channel.get {{.*}}@chan_cascade

#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @chan_cascade [2, 1] {channel_type = "npu_cascade"}
  func.func @matvec_cascade_bf16(%arg0: memref<2x128xbf16>, %arg1: memref<128xbf16>, %arg2: memref<2xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<2x128xbf16>, memref<128xbf16>, memref<2xbf16> attributes {id = 3 : i32} {
      %1 = air.segment @matvec_cascade_0 async  args(%arg10=%arg3, %arg11=%arg7, %arg12=%arg8, %arg13=%arg9) : index, memref<2x128xbf16>, memref<128xbf16>, memref<2xbf16> attributes {id = 2 : i32} {
        %c128 = arith.constant 128 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %2 = affine.apply #map()[%arg10]
        %async_token, %results = air.execute -> (memref<2x1x128xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x1x128xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<2x1x128xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_1, %results_2 = air.execute -> (memref<2x1xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x1xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<2x1xbf16, 1 : i32>
        } {id = 2 : i32}
        %3 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg11[%c0, %2, %c0] [%c2, %c1_0, %c128] [%c128, %c128, %c1_0]) {id = 1 : i32} : (memref<2x1x128xbf16, 1 : i32>, memref<2x128xbf16>)
        %async_token_3, %results_4 = air.execute -> (memref<1x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1x64xbf16, 2 : i32>
        } {id = 3 : i32}
        %async_token_5, %results_6 = air.execute -> (memref<64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64xbf16, 2 : i32>
        } {id = 4 : i32}
        %async_token_7, %results_8 = air.execute -> (memref<1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1xbf16, 2 : i32>
        } {id = 5 : i32}
        %async_token_9, %results_10 = air.execute -> (memref<16xf32, 2 : i32>) {
          %alloc = memref.alloc() : memref<16xf32, 2 : i32>
          air.execute_terminator %alloc : memref<16xf32, 2 : i32>
        } {id = 6 : i32}
        %async_token_11, %results_12 = air.execute -> (memref<16xf32, 2 : i32>) {
          %alloc = memref.alloc() : memref<16xf32, 2 : i32>
          air.execute_terminator %alloc : memref<16xf32, 2 : i32>
        } {id = 7 : i32}
        %4 = air.herd @herd_0 async [%async_token_1, %3, %async_token_3, %async_token_5, %async_token_7, %async_token_9, %async_token_11]  tile (%arg14, %arg15) in (%arg16=%c2, %arg17=%c2) args(%arg18=%results_4, %arg19=%results_6, %arg20=%results_8, %arg21=%results_10, %arg22=%results_12, %arg23=%results, %arg24=%arg12, %arg25=%results_2) : memref<1x64xbf16, 2 : i32>, memref<64xbf16, 2 : i32>, memref<1xbf16, 2 : i32>, memref<16xf32, 2 : i32>, memref<16xf32, 2 : i32>, memref<2x1x128xbf16, 1 : i32>, memref<128xbf16>, memref<2x1xbf16, 1 : i32> attributes {id = 1 : i32} {
          %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
          %c16 = arith.constant 16 : index
          %c128_20 = arith.constant 128 : index
          %cst_21 = arith.constant 0.000000e+00 : f32
          %cst_22 = arith.constant 0.000000e+00 : bf16
          %c64 = arith.constant 64 : index
          %c0_23 = arith.constant 0 : index
          %c1_24 = arith.constant 1 : index
          %6 = affine.apply #map1()[%arg15]
          %7 = affine.if #set()[%arg14, %arg15] -> !air.async.token {
            %13 = air.dma_memcpy_nd async (%arg19[] [] [], %arg24[%6] [%c64] [%c1_24]) {broadcast_set = #set, id = 2 : i32} : (memref<64xbf16, 2 : i32>, memref<128xbf16>)
            affine.yield %13 : !air.async.token
          } else {
            %13 = air.dma_memcpy_nd async (%arg19[] [] [], %arg24[%6] [%c64] [%c1_24]) {broadcast_set = #set1, id = 3 : i32} : (memref<64xbf16, 2 : i32>, memref<128xbf16>)
            affine.yield %13 : !air.async.token
          }
          %async_token_25, %results_26 = air.execute -> (memref<16xf32, 2 : i32>) {
            %alloc = memref.alloc() : memref<16xf32, 2 : i32>
            air.execute_terminator %alloc : memref<16xf32, 2 : i32>
          } {id = 8 : i32}
          %8 = air.wait_all async [%7, %async_token_25]  {id = 19 : i32}
          %9 = scf.for %arg26 = %c0_23 to %c1_24 step %c1_24 iter_args(%arg27 = %8) -> (!air.async.token) {
            %13 = air.dma_memcpy_nd async [%arg27] (%arg18[] [] [], %arg23[%arg14, %arg26, %6] [%c1_24, %c1_24, %c64] [%c128_20, %c128_20, %c1_24]) {id = 4 : i32} : (memref<1x64xbf16, 2 : i32>, memref<2x1x128xbf16, 1 : i32>)
            %14 = arith.cmpi eq, %arg15, %c1_24 : index
            %15 = air.wait_all async [%arg27, %13, %arg27, %arg27, %13, %arg27]  {id = 16 : i32}
            %16 = scf.if %14 -> (!air.async.token) {
              %18 = air.wait_all async [%arg27, %13, %arg27]  {id = 3 : i32}
              %19 = scf.for %arg28 = %c0_23 to %c1_24 step %c1_24 iter_args(%arg29 = %18) -> (!air.async.token) {
                %async_token_28 = air.execute [%arg29] {
                  vector.transfer_write %cst, %results_26[%c0_23] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2 : i32>
                } {id = 9 : i32}
                %23 = air.wait_all async [%arg29, %arg29, %async_token_28]  {id = 1 : i32}
                %24 = scf.for %arg30 = %c0_23 to %c64 step %c16 iter_args(%arg31 = %23) -> (!air.async.token) {
                  %subview_32 = memref.subview %arg18[%arg28, %arg30] [1, 16] [1, 1] : memref<1x64xbf16, 2 : i32> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>
                  %subview_33 = memref.subview %arg19[%arg30] [16] [1] : memref<64xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
                  %async_token_34, %results_35 = air.execute [%arg31] -> (vector<16xbf16>) {
                    %31 = vector.transfer_read %subview_32[%c0_23, %c0_23], %cst_22 {in_bounds = [true]} : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>, vector<16xbf16>
                    air.execute_terminator %31 : vector<16xbf16>
                  } {id = 10 : i32}
                  %async_token_36, %results_37 = air.execute [%arg31] -> (vector<16xbf16>) {
                    %31 = vector.transfer_read %subview_33[%c0_23], %cst_22 {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
                    air.execute_terminator %31 : vector<16xbf16>
                  } {id = 11 : i32}
                  %27 = arith.extf %results_35 : vector<16xbf16> to vector<16xf32>
                  %28 = arith.extf %results_37 : vector<16xbf16> to vector<16xf32>
                  %async_token_38, %results_39 = air.execute [%arg31] -> (vector<16xf32>) {
                    %31 = vector.transfer_read %results_26[%c0_23], %cst_21 {in_bounds = [true]} : memref<16xf32, 2 : i32>, vector<16xf32>
                    air.execute_terminator %31 : vector<16xf32>
                  } {id = 12 : i32}
                  %29 = vector.fma %27, %28, %results_39 : vector<16xf32>
                  %async_token_40 = air.execute [%arg31, %async_token_38] {
                    vector.transfer_write %29, %results_26[%c0_23] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2 : i32>
                  } {id = 13 : i32}
                  %30 = air.wait_all async [%async_token_34, %async_token_36, %async_token_40]  {id = 2 : i32}
                  scf.yield %30 : !air.async.token
                }
                %async_token_29, %results_30 = air.execute [%arg29, %24] -> (vector<16xf32>) {
                  %27 = vector.transfer_read %results_26[%c0_23], %cst_21 {in_bounds = [true]} : memref<16xf32, 2 : i32>, vector<16xf32>
                  air.execute_terminator %27 : vector<16xf32>
                } {id = 14 : i32}
                %25 = vector.reduction <add>, %results_30 : vector<16xf32> into f32
                %subview = memref.subview %arg21[%arg28] [1] [1] : memref<16xf32, 2 : i32> to memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                %async_token_31 = air.execute [%arg29] {
                  memref.store %25, %subview[%c0_23] : memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                } {id = 15 : i32}
                %26 = air.wait_all async [%async_token_29, %async_token_31]  {id = 4 : i32}
                scf.yield %26 : !air.async.token
              }
              %20 = arith.subi %arg15, %c1_24 : index
              %21 = air.channel.put async  @chan_cascade[%arg14, %20] (%arg21[] [] []) {id = 5 : i32} : (memref<16xf32, 2 : i32>)
              %22 = air.wait_all async [%19, %21]  {id = 17 : i32}
              scf.yield %22 : !air.async.token
            } else {
              %18 = arith.cmpi eq, %arg15, %c0_23 : index
              %19 = air.wait_all async [%arg27, %13, %arg27, %arg27, %13, %arg27]  {id = 13 : i32}
              %20 = scf.if %18 -> (!air.async.token) {
                %22 = air.channel.get async  @chan_cascade[%arg14, %arg15] (%arg22[] [] []) {id = 6 : i32} : (memref<16xf32, 2 : i32>)
                %23 = air.wait_all async [%arg27, %13, %arg27, %22]  {id = 7 : i32}
                %24 = scf.for %arg28 = %c0_23 to %c1_24 step %c1_24 iter_args(%arg29 = %23) -> (!air.async.token) {
                  %async_token_28 = air.execute [%arg29] {
                    vector.transfer_write %cst, %results_26[%c0_23] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2 : i32>
                  } {id = 16 : i32}
                  %26 = air.wait_all async [%arg29, %arg29, %async_token_28]  {id = 5 : i32}
                  %27 = scf.for %arg30 = %c0_23 to %c64 step %c16 iter_args(%arg31 = %26) -> (!air.async.token) {
                    %subview_35 = memref.subview %arg18[%arg28, %arg30] [1, 16] [1, 1] : memref<1x64xbf16, 2 : i32> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>
                    %subview_36 = memref.subview %arg19[%arg30] [16] [1] : memref<64xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
                    %async_token_37, %results_38 = air.execute [%arg31] -> (vector<16xbf16>) {
                      %37 = vector.transfer_read %subview_35[%c0_23, %c0_23], %cst_22 {in_bounds = [true]} : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>, vector<16xbf16>
                      air.execute_terminator %37 : vector<16xbf16>
                    } {id = 17 : i32}
                    %async_token_39, %results_40 = air.execute [%arg31] -> (vector<16xbf16>) {
                      %37 = vector.transfer_read %subview_36[%c0_23], %cst_22 {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
                      air.execute_terminator %37 : vector<16xbf16>
                    } {id = 18 : i32}
                    %33 = arith.extf %results_38 : vector<16xbf16> to vector<16xf32>
                    %34 = arith.extf %results_40 : vector<16xbf16> to vector<16xf32>
                    %async_token_41, %results_42 = air.execute [%arg31] -> (vector<16xf32>) {
                      %37 = vector.transfer_read %results_26[%c0_23], %cst_21 {in_bounds = [true]} : memref<16xf32, 2 : i32>, vector<16xf32>
                      air.execute_terminator %37 : vector<16xf32>
                    } {id = 19 : i32}
                    %35 = vector.fma %33, %34, %results_42 : vector<16xf32>
                    %async_token_43 = air.execute [%arg31, %async_token_41] {
                      vector.transfer_write %35, %results_26[%c0_23] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2 : i32>
                    } {id = 20 : i32}
                    %36 = air.wait_all async [%async_token_37, %async_token_39, %async_token_43]  {id = 6 : i32}
                    scf.yield %36 : !air.async.token
                  }
                  %async_token_29, %results_30 = air.execute [%arg29, %27] -> (vector<16xf32>) {
                    %33 = vector.transfer_read %results_26[%c0_23], %cst_21 {in_bounds = [true]} : memref<16xf32, 2 : i32>, vector<16xf32>
                    air.execute_terminator %33 : vector<16xf32>
                  } {id = 21 : i32}
                  %28 = vector.reduction <add>, %results_30 : vector<16xf32> into f32
                  %subview = memref.subview %arg22[%arg28] [1] [1] : memref<16xf32, 2 : i32> to memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                  %async_token_31, %results_32 = air.execute [%arg29] -> (f32) {
                    %33 = memref.load %subview[%c0_23] : memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                    air.execute_terminator %33 : f32
                  } {id = 22 : i32}
                  %29 = arith.addf %results_32, %28 : f32
                  %30 = arith.truncf %29 : f32 to bf16
                  %31 = affine.apply #map2()[%arg26, %arg28]
                  %subview_33 = memref.subview %arg20[%31] [1] [1] : memref<1xbf16, 2 : i32> to memref<1xbf16, strided<[1], offset: ?>, 2 : i32>
                  %async_token_34 = air.execute [%arg29] {
                    memref.store %30, %subview_33[%c0_23] : memref<1xbf16, strided<[1], offset: ?>, 2 : i32>
                  } {id = 23 : i32}
                  %32 = air.wait_all async [%async_token_29, %async_token_31, %async_token_34]  {id = 8 : i32}
                  scf.yield %32 : !air.async.token
                }
                %25 = air.wait_all async [%24]  {id = 14 : i32}
                scf.yield %25 : !air.async.token
              } else {
                %22 = air.channel.get async  @chan_cascade[%arg14, %arg15] (%arg22[] [] []) {id = 7 : i32} : (memref<16xf32, 2 : i32>)
                %23 = air.wait_all async [%arg27, %13, %arg27, %22]  {id = 11 : i32}
                %24 = scf.for %arg28 = %c0_23 to %c1_24 step %c1_24 iter_args(%arg29 = %23) -> (!air.async.token) {
                  %async_token_28 = air.execute [%arg29] {
                    vector.transfer_write %cst, %results_26[%c0_23] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2 : i32>
                  } {id = 24 : i32}
                  %28 = air.wait_all async [%arg29, %arg29, %async_token_28]  {id = 9 : i32}
                  %29 = scf.for %arg30 = %c0_23 to %c64 step %c16 iter_args(%arg31 = %28) -> (!air.async.token) {
                    %subview_35 = memref.subview %arg18[%arg28, %arg30] [1, 16] [1, 1] : memref<1x64xbf16, 2 : i32> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>
                    %subview_36 = memref.subview %arg19[%arg30] [16] [1] : memref<64xbf16, 2 : i32> to memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
                    %async_token_37, %results_38 = air.execute [%arg31] -> (vector<16xbf16>) {
                      %37 = vector.transfer_read %subview_35[%c0_23, %c0_23], %cst_22 {in_bounds = [true]} : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>, vector<16xbf16>
                      air.execute_terminator %37 : vector<16xbf16>
                    } {id = 25 : i32}
                    %async_token_39, %results_40 = air.execute [%arg31] -> (vector<16xbf16>) {
                      %37 = vector.transfer_read %subview_36[%c0_23], %cst_22 {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
                      air.execute_terminator %37 : vector<16xbf16>
                    } {id = 26 : i32}
                    %33 = arith.extf %results_38 : vector<16xbf16> to vector<16xf32>
                    %34 = arith.extf %results_40 : vector<16xbf16> to vector<16xf32>
                    %async_token_41, %results_42 = air.execute [%arg31] -> (vector<16xf32>) {
                      %37 = vector.transfer_read %results_26[%c0_23], %cst_21 {in_bounds = [true]} : memref<16xf32, 2 : i32>, vector<16xf32>
                      air.execute_terminator %37 : vector<16xf32>
                    } {id = 27 : i32}
                    %35 = vector.fma %33, %34, %results_42 : vector<16xf32>
                    %async_token_43 = air.execute [%arg31, %async_token_41] {
                      vector.transfer_write %35, %results_26[%c0_23] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2 : i32>
                    } {id = 28 : i32}
                    %36 = air.wait_all async [%async_token_37, %async_token_39, %async_token_43]  {id = 10 : i32}
                    scf.yield %36 : !air.async.token
                  }
                  %async_token_29, %results_30 = air.execute [%arg29, %29] -> (vector<16xf32>) {
                    %33 = vector.transfer_read %results_26[%c0_23], %cst_21 {in_bounds = [true]} : memref<16xf32, 2 : i32>, vector<16xf32>
                    air.execute_terminator %33 : vector<16xf32>
                  } {id = 29 : i32}
                  %30 = vector.reduction <add>, %results_30 : vector<16xf32> into f32
                  %subview = memref.subview %arg22[%arg28] [1] [1] : memref<16xf32, 2 : i32> to memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                  %async_token_31, %results_32 = air.execute [%arg29] -> (f32) {
                    %33 = memref.load %subview[%c0_23] : memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                    air.execute_terminator %33 : f32
                  } {id = 30 : i32}
                  %31 = arith.addf %results_32, %30 : f32
                  %subview_33 = memref.subview %arg21[%arg28] [1] [1] : memref<16xf32, 2 : i32> to memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                  %async_token_34 = air.execute [%arg29] {
                    memref.store %31, %subview_33[%c0_23] : memref<1xf32, strided<[1], offset: ?>, 2 : i32>
                  } {id = 31 : i32}
                  %32 = air.wait_all async [%async_token_29, %async_token_31, %async_token_34]  {id = 12 : i32}
                  scf.yield %32 : !air.async.token
                }
                %25 = arith.subi %arg15, %c1_24 : index
                %26 = air.channel.put async  @chan_cascade[%arg14, %25] (%arg21[] [] []) {id = 8 : i32} : (memref<16xf32, 2 : i32>)
                %27 = air.wait_all async [%24, %26]  {id = 15 : i32}
                scf.yield %27 : !air.async.token
              }
              %21 = air.wait_all async [%19]  {id = 18 : i32}
              scf.yield %21 : !air.async.token
            }
            %17 = air.wait_all async [%15]  {id = 20 : i32}
            scf.yield %17 : !air.async.token
          }
          %10 = arith.cmpi eq, %arg15, %c0_23 : index
          %11 = air.wait_all async  {id = 21 : i32}
          %12 = scf.if %10 -> (!air.async.token) {
            %13 = affine.if #set2()[%arg14, %arg15] -> !air.async.token {
              %15 = air.dma_memcpy_nd async (%arg25[%c0_23, %c0_23] [%c1_24, %c1_24] [%c1_24, %c1_24], %arg20[] [%c1_24] [%c1_24]) {broadcast_set = #set2, id = 9 : i32} : (memref<2x1xbf16, 1 : i32>, memref<1xbf16, 2 : i32>)
              affine.yield %15 : !air.async.token
            } else {
              %15 = air.dma_memcpy_nd async (%arg25[%c1_24, %c0_23] [%c1_24, %c1_24] [%c1_24, %c1_24], %arg20[] [%c1_24] [%c1_24]) {broadcast_set = #set3, id = 10 : i32} : (memref<2x1xbf16, 1 : i32>, memref<1xbf16, 2 : i32>)
              affine.yield %15 : !air.async.token
            }
            %14 = air.wait_all async [%13]  {id = 22 : i32}
            scf.yield %14 : !air.async.token
          } else {
            %13 = air.wait_all async  {id = 23 : i32}
            scf.yield %13 : !air.async.token
          }
          %async_token_27 = air.execute {
            memref.dealloc %results_26 : memref<16xf32, 2 : i32>
          } {id = 32 : i32}
        }
        %5 = air.dma_memcpy_nd async [%4] (%arg13[%2] [%c2] [%c1_0], %results_2[%c0, %c0] [%c2, %c1_0] [%c1_0, %c1_0]) {id = 11 : i32} : (memref<2xbf16>, memref<2x1xbf16, 1 : i32>)
        %async_token_13 = air.execute [%4] {
          memref.dealloc %results : memref<2x1x128xbf16, 1 : i32>
        } {id = 33 : i32}
        %async_token_14 = air.execute [%5] {
          memref.dealloc %results_2 : memref<2x1xbf16, 1 : i32>
        } {id = 34 : i32}
        %async_token_15 = air.execute [%4] {
          memref.dealloc %results_4 : memref<1x64xbf16, 2 : i32>
        } {id = 35 : i32}
        %async_token_16 = air.execute [%4] {
          memref.dealloc %results_6 : memref<64xbf16, 2 : i32>
        } {id = 36 : i32}
        %async_token_17 = air.execute [%4] {
          memref.dealloc %results_8 : memref<1xbf16, 2 : i32>
        } {id = 37 : i32}
        %async_token_18 = air.execute [%4] {
          memref.dealloc %results_10 : memref<16xf32, 2 : i32>
        } {id = 38 : i32}
        %async_token_19 = air.execute [%4] {
          memref.dealloc %results_12 : memref<16xf32, 2 : i32>
        } {id = 39 : i32}
      }
    }
    return
  }
}
