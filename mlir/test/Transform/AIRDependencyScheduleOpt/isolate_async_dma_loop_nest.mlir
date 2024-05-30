//===- isolate_async_dma_loop_nest.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests --split-input-file | FileCheck %s

// Isolate scf for loops containing dma ops into perfectly nested loop.

// CHECK-LABEL: func0

// CHECK: air.launch
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: scf.yield

// CHECK: air.segment @segment_0
// CHECK: scf.for
// CHECK: scf.parallel
// CHECK: air.channel.put{{.*}}@channel_2
// CHECK: scf.reduce
// CHECK: scf.yield

// CHECK: scf.for
// CHECK: scf.parallel
// CHECK: air.channel.get{{.*}}@channel_3
// CHECK: scf.reduce
// CHECK: scf.yield

// CHECK: air.herd @herd_0
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: air.herd_terminator

// CHECK: %[[EVENT0:.*]]:4 = scf.for
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: scf.yield

// CHECK: air.segment_terminator

// CHECK: air.launch_terminator

module {
  air.channel @channel_3 [2, 2]
  air.channel @channel_2 [2, 2]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @func0(%extArg0: memref<32x32xi32>, %extArg1: memref<32x32xi32>, %extArg2: memref<32x32xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) args(%arg4=%extArg2, %arg5=%extArg0, %arg6=%extArg1) : memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32> attributes {id = 1 : i32} {
      %c8 = arith.constant 8 : index
      %c0_8 = arith.constant 0 : index
      %c1_9 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %2 = air.wait_all async 
      %3 = scf.for %arg7 = %c0_8 to %c32 step %c8 iter_args(%arg8 = %2) -> (!air.async.token) {
        %6 = air.channel.put async [%arg8]  @channel_1[] (%arg5[] [] []) {id = 2 : i32} : (memref<32x32xi32>)
        %7 = air.channel.put async [%6]  @channel_1[] (%arg6[] [] []) {id = 3 : i32} : (memref<32x32xi32>)
        scf.yield %7 : !air.async.token
      }
      %5 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c16 = arith.constant 16 : index
        %c1_22 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0_23 = arith.constant 0 : index
        %c32_24 = arith.constant 32 : index
        %c8_25 = arith.constant 8 : index
        %6 = air.wait_all async 
        %async_token_26, %results_27 = air.execute -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %8 = scf.for %arg7 = %c0_23 to %c32_24 step %c8_25 iter_args(%arg8 = %6) -> (!air.async.token) {
          %11 = air.herd @herd_0 async [%arg8]  tile (%arg9, %arg10) in (%arg11=%c2, %arg12=%c2) attributes {id = 3 : i32} {
            %async_token_37, %results_38 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %15 = air.channel.get async [%async_token_37]  @channel_0[%arg9, %arg10] (%results_38[] [] []) {id = 14 : i32} : (memref<32x32xi32, 2>)
            %async_token_41, %results_42 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %17 = air.channel.get async [%async_token_41]  @channel_2[%arg9, %arg10] (%results_42[] [] []) {id = 18 : i32} : (memref<32x32xi32, 2>)
            %async_token_43 = air.wait_all async [%15, %17]
            %18 = air.channel.put async [%async_token_43]  @channel_3[%arg9, %arg10] (%results_42[] [] []) {id = 19 : i32} : (memref<32x32xi32, 2>)
            %async_token_44 = air.execute [%async_token_43] {
              memref.dealloc %results_38 : memref<32x32xi32, 2>
            }
            %async_token_46 = air.execute [%18] {
              memref.dealloc %results_42 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          %12 = scf.parallel (%arg9, %arg10) = (%c0_23, %c0_23) to (%c2, %c2) step (%c1_22, %c1_22) init (%arg8) -> !air.async.token {
            %15 = air.channel.put async [%arg8]  @channel_2[%arg9, %arg10] (%results_27[] [] []) {id = 12 : i32} : (memref<32x32xi32, 1>)
            scf.reduce(%15 : !air.async.token) {
            ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
              %16 = air.wait_all async [%arg11, %arg12] 
              scf.reduce.return %16 : !air.async.token
            }
          }
          %13 = scf.parallel (%arg9, %arg10) = (%c0_23, %c0_23) to (%c2, %c2) step (%c1_22, %c1_22) init (%arg8) -> !air.async.token {
            %15 = air.channel.get async [%arg8]  @channel_3[%arg9, %arg10] (%results_27[] [] []) {id = 13 : i32} : (memref<32x32xi32, 1>)
            scf.reduce(%15 : !air.async.token) {
            ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
              %16 = air.wait_all async [%arg11, %arg12] 
              scf.reduce.return %16 : !air.async.token
            }
          }
          %14 = air.wait_all async [%11, %12, %13] 
          scf.yield %14 : !air.async.token
        } {unroll = 4 : i32}
        %async_token_28, %results_29 = air.execute [%6] -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %async_token_30, %results_31 = air.execute [%async_token_28] -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %async_token_32, %results_33 = air.execute [%async_token_30] -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %async_token_34, %results_35 = air.execute [%async_token_30] -> (memref<32x32xi32, 1>) {
          %alloc = memref.alloc() : memref<32x32xi32, 1>
          air.execute_terminator %alloc : memref<32x32xi32, 1>
        }
        %9:4 = scf.for %arg7 = %c0_23 to %c32_24 step %c16 iter_args(%arg8 = %async_token_32, %arg9 = %async_token_34, %arg10 = %async_token_34, %arg11 = %async_token_34) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %11 = air.wait_all async 
          %12 = air.channel.get async [%arg11, %async_token_32, %arg8]  @channel_1[] (%results_33[] [] []) {id = 6 : i32} : (memref<32x32xi32, 1>)
          %13 = air.channel.get async [%12, %arg11, %async_token_34, %arg8]  @channel_1[] (%results_35[] [] []) {id = 7 : i32} : (memref<32x32xi32, 1>)
          %14 = air.channel.put async [%arg10, %11]  @channel_0[] (%results_33[] [] []) {id = 8 : i32} : (memref<32x32xi32, 1>)
          %async_token_37 = air.execute {
            memref.dealloc %results_33 : memref<32x32xi32, 1>
          }
          %async_token_38 = air.execute {
            memref.dealloc %results_35 : memref<32x32xi32, 1>
          }
          %18 = air.wait_all async 
          %19 = air.channel.get async [%13, %11, %arg9]  @channel_1[] (%results_31[] [] []) {id = 6 : i32} : (memref<32x32xi32, 1>)
          %20 = air.channel.get async [%19, %13, %11, %arg9]  @channel_1[] (%results_29[] [] []) {id = 7 : i32} : (memref<32x32xi32, 1>)
          %21 = air.channel.put async [%13, %18]  @channel_0[] (%results_31[] [] []) {id = 8 : i32} : (memref<32x32xi32, 1>)
          %async_token_39 = air.execute {
            memref.dealloc %results_31 : memref<32x32xi32, 1>
          }
          %async_token_40 = air.execute {
            memref.dealloc %results_29 : memref<32x32xi32, 1>
          }
          scf.yield %13, %20, %20, %20 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        } {unroll = 2 : i32}
        %10 = air.wait_all async [%8, %9#1]
        %async_token_36 = air.execute [%10] {
          memref.dealloc %results_27 : memref<32x32xi32, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Check air.herd and air.channel op hoisting.

// CHECK-LABEL: func1

// CHECK: air.launch
// CHECK: air.segment @segment_0
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST64:.*]] = arith.constant 64 : index
// CHECK: scf.for {{.*}} = %[[CST0]] to %[[CST64]] step %[[CST64]] iter_args
// CHECK: scf.parallel
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: scf.reduce
// CHECK: scf.yield

// CHECK: air.herd @herd_0
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[CST512:.*]] = arith.constant 512 : index
// CHECK: linalg.fill
// CHECK: scf.for {{.*}} = %[[CST0]] to %[[CST512]] step %[[CST64]] iter_args
// CHECK: scf.yield
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_0 [2, 2]
  func.func @func1() {
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c32, %arg3=%c32) attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c32_0 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %async_token, %results = air.execute -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %2 = scf.for %arg4 = %c0 to %c512 step %c64 iter_args(%arg5 = %async_token) -> (!air.async.token) {
          %3 = air.herd @herd_0 async [%arg5]  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) attributes {id = 3 : i32} {
            %c0_i32 = arith.constant 0 : i32
            %async_token_1, %results_2 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_3 = air.execute [%async_token_1] {
              linalg.fill ins(%c0_i32 : i32) outs(%results_2 : memref<32x32xi32, 2>)
            }
            %6 = air.channel.put async [%async_token_3]  @channel_0[%arg6, %arg7] (%results_2[] [] []) {id = 15 : i32} : (memref<32x32xi32, 2>)
            %async_token_4 = air.execute [%6] {
              memref.dealloc %results_2 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          %4 = scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%arg5) -> !air.async.token {
            %async_token_1, %results_2 = air.execute -> (index) {
              %7 = affine.apply #map()[%arg6]
              air.execute_terminator %7 : index
            }
            %async_token_3, %results_4 = air.execute -> (index) {
              %7 = affine.apply #map()[%arg7]
              air.execute_terminator %7 : index
            }
            %6 = air.channel.get async [%arg5, %async_token_3, %async_token_1]  @channel_0[%arg6, %arg7] (%results[%results_2, %results_4] [%c32_0, %c32_0] [%c64, %c1]) {id = 10 : i32} : (memref<64x64xi32, 1>)
            scf.reduce(%6 : !air.async.token) {
            ^bb0(%arg8: !air.async.token, %arg9: !air.async.token):
              %7 = air.wait_all async [%arg8, %arg9] 
              scf.reduce.return %7 : !air.async.token
            }
          }
          %5 = air.wait_all async [%3, %4] 
          scf.yield %5 : !air.async.token
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Loop nest.

// CHECK-LABEL: func2

// CHECK: air.launch
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0

// CHECK: scf.for %{{.*}} = %c0 to %c2048 step %c256 iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK: scf.for %{{.*}} = %c0 to %c256 step %c64 iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
// CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !air.async.token, !air.async.token, !air.async.token, !air.async.token
// CHECK: scf.yield %{{.*}} : !air.async.token

// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

module {
  func.func @func2() {
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %2 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) attributes {id = 3 : i32} {
          %c64 = arith.constant 64 : index
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c2048 = arith.constant 2048 : index
          %async_token_1, %results_2 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %3 = scf.for %arg11 = %c0 to %c2048 step %c256 iter_args(%arg12 = %async_token_1) -> (!air.async.token) {
            %async_token_4, %results_5 = air.execute [%arg12] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_6, %results_7 = air.execute [%async_token_4] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_8, %results_9 = air.execute [%async_token_6] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_10, %results_11 = air.execute [%async_token_6] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %5:4 = scf.for %arg13 = %c0 to %c256 step %c64 iter_args(%arg14 = %async_token_8, %arg15 = %async_token_10, %arg16 = %async_token_10, %arg17 = %async_token_10) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
              %6 = air.channel.get async [%arg17, %arg14, %async_token_8]  @channel_2[%arg7, %arg8] (%results_9[] [] []) {id = 9 : i32} : (memref<32x32xi32, 2>)
              %7 = air.channel.get async [%arg17, %arg14, %async_token_10]  @channel_3[%arg7, %arg8] (%results_11[] [] []) {id = 10 : i32} : (memref<32x32xi32, 2>)
              %async_token_12 = air.wait_all async [%arg16, %7, %6]
              %async_token_13 = air.execute {
                memref.dealloc %results_9 : memref<32x32xi32, 2>
              }
              %async_token_14 = air.execute {
                memref.dealloc %results_11 : memref<32x32xi32, 2>
              }
              %8 = air.channel.get async [%7, %6, %arg15]  @channel_2[%arg7, %arg8] (%results_7[] [] []) {id = 9 : i32} : (memref<32x32xi32, 2>)
              %9 = air.channel.get async [%7, %6, %arg15]  @channel_3[%arg7, %arg8] (%results_5[] [] []) {id = 10 : i32} : (memref<32x32xi32, 2>)
              %async_token_15 = air.wait_all async [%async_token_12, %9, %8]
              %async_token_16 = air.execute {
                memref.dealloc %results_7 : memref<32x32xi32, 2>
              }
              %async_token_17 = air.execute {
                memref.dealloc %results_5 : memref<32x32xi32, 2>
              }
              %10 = air.wait_all async [%8, %9] 
              scf.yield %async_token_12, %async_token_15, %async_token_15, %10 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
            }
            scf.yield %5#1 : !air.async.token
          }
          %4 = air.channel.put async [%3]  @channel_4[%arg7, %arg8] (%results_2[] [] []) {id = 11 : i32} : (memref<32x32xi32, 2>)
          %async_token_3 = air.execute [%4] {
            memref.dealloc %results_2 : memref<32x32xi32, 2>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL: func3

// CHECK: air.launch

// CHECK: scf.for %{{.*}} = %c0 to %c1024 step %c256 iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put
// CHECK: scf.for %{{.*}} = %c0 to %c1024 step %c256 iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put
// CHECK: scf.for %{{.*}} = %c0 to %c1024 step %c256 iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put
// CHECK: scf.for %{{.*}} = %c0 to %c1024 step %c256 iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put

// CHECK: air.segment @segment_0

// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[CST1024:.*]] = arith.constant 1024 : index

// CHECK: scf.for %{{.*}} = %[[CST0]] to %[[CST1024]] step %[[CST256]] iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: air.channel.get

// CHECK: scf.for %{{.*}} = %[[CST0]] to %[[CST1024]] step %[[CST256]] iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: air.channel.get

// CHECK: scf.for %{{.*}} = %[[CST0]] to %[[CST1024]] step %[[CST256]] iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: air.channel.get

// CHECK: scf.for %{{.*}} = %[[CST0]] to %[[CST1024]] step %[[CST256]] iter_args(%{{.*}} = %{{.*}}) -> (!air.async.token) {
// CHECK-NEXT: air.channel.get

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 256 + 64)>
#map2 = affine_map<()[s0] -> (s0 * 256 + 128)>
#map3 = affine_map<()[s0] -> (s0 * 256 + 192)>
module {
  air.channel @channel_0 [4, 1]
  func.func @func3(%arg0: memref<512x1024xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c2, %arg4=%c2) args(%arg5=%arg0) : memref<512x1024xbf16> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg6 = %c0 to %c1024 step %c256 iter_args(%arg7 = %1) -> (!air.async.token) {
        %4 = affine.apply #map()[%arg1]
        %5 = air.channel.put async [%arg7]  @channel_0[%c0, %c0] (%arg5[%4, %arg6] [%c64, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<512x1024xbf16>)
        %6 = affine.apply #map1()[%arg1]
        %7 = air.channel.put async [%arg7]  @channel_0[%c1, %c0] (%arg5[%6, %arg6] [%c64, %c256] [%c1024, %c1]) {id = 2 : i32} : (memref<512x1024xbf16>)
        %8 = affine.apply #map2()[%arg1]
        %9 = air.channel.put async [%arg7]  @channel_0[%c2_0, %c0] (%arg5[%8, %arg6] [%c64, %c256] [%c1024, %c1]) {id = 3 : i32} : (memref<512x1024xbf16>)
        %10 = affine.apply #map3()[%arg1]
        %11 = air.channel.put async [%arg7]  @channel_0[%c3, %c0] (%arg5[%10, %arg6] [%c64, %c256] [%c1024, %c1]) {id = 4 : i32} : (memref<512x1024xbf16>)
        %12 = air.wait_all async [%5, %7, %9, %11] 
        scf.yield %12 : !air.async.token
      }
      %3 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c3_1 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c1_4 = arith.constant 1 : index
        %c0_5 = arith.constant 0 : index
        %c1024_6 = arith.constant 1024 : index
        %c256_7 = arith.constant 256 : index
        %async_token, %results = air.execute -> (memref<64x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x1024xbf16, 1>
          air.execute_terminator %alloc : memref<64x1024xbf16, 1>
        }
        %async_token_8, %results_9 = air.execute -> (memref<64x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x1024xbf16, 1>
          air.execute_terminator %alloc : memref<64x1024xbf16, 1>
        }
        %async_token_10, %results_11 = air.execute -> (memref<64x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x1024xbf16, 1>
          air.execute_terminator %alloc : memref<64x1024xbf16, 1>
        }
        %async_token_12, %results_13 = air.execute -> (memref<64x1024xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x1024xbf16, 1>
          air.execute_terminator %alloc : memref<64x1024xbf16, 1>
        }
        %4 = air.wait_all async 
        %5 = scf.for %arg6 = %c0_5 to %c1024_6 step %c256_7 iter_args(%arg7 = %4) -> (!air.async.token) {
          %6 = air.channel.get async [%arg7]  @channel_0[%c0_5, %c0_5] (%results[%c0_5, %arg6] [%c64_3, %c256_7] [%c1024_6, %c1_4]) {id = 13 : i32} : (memref<64x1024xbf16, 1>)
          %7 = air.channel.get async [%arg7]  @channel_0[%c1_4, %c0_5] (%results_9[%c0_5, %arg6] [%c64_3, %c256_7] [%c1024_6, %c1_4]) {id = 14 : i32} : (memref<64x1024xbf16, 1>)
          %8 = air.channel.get async [%arg7]  @channel_0[%c2_2, %c0_5] (%results_11[%c0_5, %arg6] [%c64_3, %c256_7] [%c1024_6, %c1_4]) {id = 15 : i32} : (memref<64x1024xbf16, 1>)
          %9 = air.channel.get async [%arg7]  @channel_0[%c3_1, %c0_5] (%results_13[%c0_5, %arg6] [%c64_3, %c256_7] [%c1024_6, %c1_4]) {id = 16 : i32} : (memref<64x1024xbf16, 1>)
          %10 = air.wait_all async [%6, %7, %8, %9] 
          scf.yield %10 : !air.async.token
        }
        %async_token_14 = air.execute [%5] {
          memref.dealloc %results_13 : memref<64x1024xbf16, 1>
        }
        %async_token_15 = air.execute [%5] {
          memref.dealloc %results_11 : memref<64x1024xbf16, 1>
        }
        %async_token_16 = air.execute [%5] {
          memref.dealloc %results_9 : memref<64x1024xbf16, 1>
        }
        %async_token_17 = air.execute [%5] {
          memref.dealloc %results : memref<64x1024xbf16, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
