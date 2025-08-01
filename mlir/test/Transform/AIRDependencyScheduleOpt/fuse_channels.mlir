//===- fuse_channels.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=false" --split-input-file | FileCheck %s
// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1,L2,L3" --split-input-file | FileCheck %s --check-prefix=AGGRESSIVE
// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" --split-input-file | FileCheck %s --check-prefix=AGGL1

// Have multiple channel put-get pairs share the same symbolic channels.
// CHECK-LABEL: func0
// CHECK: air.launch
// CHECK: air.channel.put @channel_0
// CHECK: air.channel.put @channel_1
// CHECK: air.segment
// CHECK: air.channel.get @channel_0
// CHECK: air.channel.get @channel_1
// AGGRESSIVE-LABEL: func0
// AGGRESSIVE: air.launch
// AGGRESSIVE: air.channel.put @channel_0
// AGGRESSIVE: air.channel.put @channel_0
// AGGRESSIVE: air.segment
// AGGRESSIVE: air.channel.get @channel_0
// AGGRESSIVE: air.channel.get @channel_0
// AGGL1-LABEL: func0
// AGGL1: air.launch
// AGGL1: air.channel.put @channel_0
// AGGL1: air.channel.put @channel_1
// AGGL1: air.segment
// AGGL1: air.channel.get @channel_0
// AGGL1: air.channel.get @channel_1

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @func0(){
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) {
      %alloc_0 = memref.alloc() : memref<4x4xi32>
      %alloc_1 = memref.alloc() : memref<4x4xi32>
      air.channel.put @channel_0[%arg3, %arg4] (%alloc_0[] [] []) : (memref<4x4xi32>)
      air.channel.put @channel_1[%arg3, %arg4] (%alloc_1[] [] []) : (memref<4x4xi32>)
      air.segment {
        %c2 = arith.constant 2 : index
        %alloc_2 = memref.alloc() : memref<4x4xi32, 1>
        %alloc_3 = memref.alloc() : memref<4x4xi32, 1>
        air.channel.get @channel_0[] (%alloc_2[] [] []) : (memref<4x4xi32, 1>)
        air.channel.get @channel_1[] (%alloc_3[] [] []) : (memref<4x4xi32, 1>)
        air.herd @herd_0 tile (%arg12, %arg13) in (%arg14=%c2, %arg15=%c2) {
        }
        memref.dealloc %alloc_2 : memref<4x4xi32, 1>
        memref.dealloc %alloc_3 : memref<4x4xi32, 1>
      }
      memref.dealloc %alloc_0 : memref<4x4xi32>
      memref.dealloc %alloc_1 : memref<4x4xi32>
    }
    return
  }
}

// -----

// CHECK-LABEL: func1
// CHECK: air.launch
// CHECK: air.segment
// CHECK: air.channel.put @channel_2
// CHECK: scf.for
// CHECK: air.channel.put @channel_0
// CHECK: air.channel.put @channel_1
// CHECK: air.herd
// CHECK: air.channel.get @channel_2
// CHECK: scf.for
// CHECK: air.channel.get @channel_0
// CHECK: air.channel.get @channel_1
// AGGRESSIVE-LABEL: func1
// AGGRESSIVE: air.launch
// AGGRESSIVE: air.segment
// AGGRESSIVE: air.channel.put @channel_2
// AGGRESSIVE: scf.for
// AGGRESSIVE: air.channel.put @channel_0
// AGGRESSIVE: air.channel.put @channel_0
// AGGRESSIVE: air.herd
// AGGRESSIVE: air.channel.get @channel_2
// AGGRESSIVE: scf.for
// AGGRESSIVE: air.channel.get @channel_0
// AGGRESSIVE: air.channel.get @channel_0
// AGGL1-LABEL: func1
// AGGL1: air.launch
// AGGL1: air.segment
// AGGL1: air.channel.put @channel_2
// AGGL1: scf.for
// AGGL1: air.channel.put @channel_0
// AGGL1: air.channel.put @channel_0
// AGGL1: air.herd
// AGGL1: air.channel.get @channel_2
// AGGL1: scf.for
// AGGL1: air.channel.get @channel_0
// AGGL1: air.channel.get @channel_0

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  func.func @func1(){
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) {
      air.segment {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        %alloc_1 = memref.alloc() : memref<4x4xi32, 1>
        %alloc_2 = memref.alloc() : memref<4x4xi32, 1>
        %alloc_3 = memref.alloc() : memref<4x4xi32, 1>
        air.channel.put @channel_2[] (%alloc_1[] [] []) : (memref<4x4xi32, 1>)
        scf.for %arg0 = %c0 to %c2 step %c1_1 {
          air.channel.put @channel_0[] (%alloc_2[] [] []) : (memref<4x4xi32, 1>)
        }
        scf.for %arg0 = %c0 to %c2 step %c1_1 {
          air.channel.put @channel_1[] (%alloc_3[] [] []) : (memref<4x4xi32, 1>)
        }
        memref.dealloc %alloc_1 : memref<4x4xi32, 1>
        memref.dealloc %alloc_2 : memref<4x4xi32, 1>
        memref.dealloc %alloc_3 : memref<4x4xi32, 1>
        air.herd @herd_0 tile (%arg12, %arg13) in (%arg14=%c2, %arg15=%c2) {
          %c0_2 = arith.constant 0 : index
          %c2_2 = arith.constant 2 : index
          %c1_2 = arith.constant 1 : index
          %alloc_4 = memref.alloc() : memref<4x4xi32, 2>
          air.channel.get @channel_2[] (%alloc_4[] [] []) : (memref<4x4xi32, 2>)
          scf.for %arg0 = %c0_2 to %c2_2 step %c1_2 {
            %alloc_5 = memref.alloc() : memref<4x4xi32, 2>
            %alloc_6 = memref.alloc() : memref<4x4xi32, 2>
            air.channel.get @channel_0[] (%alloc_5[] [] []) : (memref<4x4xi32, 2>)
            air.channel.get @channel_1[] (%alloc_6[] [] []) : (memref<4x4xi32, 2>)
            memref.dealloc %alloc_5 : memref<4x4xi32, 2>
            memref.dealloc %alloc_6 : memref<4x4xi32, 2>
          }
          memref.dealloc %alloc_4 : memref<4x4xi32, 2>
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func2
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.segment
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.channel.get{{.*}}@channel_1
// AGGRESSIVE-LABEL: func2
// AGGRESSIVE: air.launch
// AGGRESSIVE: air.channel.put{{.*}}@channel_0
// AGGRESSIVE: air.channel.put{{.*}}@channel_0
// AGGRESSIVE: air.segment
// AGGRESSIVE: air.channel.get{{.*}}@channel_0
// AGGRESSIVE: air.channel.get{{.*}}@channel_0
// AGGL1-LABEL: func2
// AGGL1: air.launch
// AGGL1: air.channel.put{{.*}}@channel_0
// AGGL1: air.channel.put{{.*}}@channel_1
// AGGL1: air.segment
// AGGL1: air.channel.get{{.*}}@channel_0
// AGGL1: air.channel.get{{.*}}@channel_1

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @func2() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %async_token, %results = air.execute -> (memref<4x4xi32>) {
        %alloc = memref.alloc() : memref<4x4xi32>
        air.execute_terminator %alloc : memref<4x4xi32>
      }
      %async_token_0, %results_1 = air.execute -> (memref<4x4xi32>) {
        %alloc = memref.alloc() : memref<4x4xi32>
        air.execute_terminator %alloc : memref<4x4xi32>
      }
      %1 = air.channel.put async [%async_token]  @channel_0[%arg0, %arg1] (%results[] [] []) {id = 1 : i32} : (memref<4x4xi32>)
      %2 = air.channel.put async [%async_token_0]  @channel_1[%arg0, %arg1] (%results_1[] [] []) {id = 2 : i32} : (memref<4x4xi32>)
      %3 = air.segment async  attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %async_token_4, %results_5 = air.execute -> (memref<4x4xi32, 1>) {
          %alloc = memref.alloc() : memref<4x4xi32, 1>
          air.execute_terminator %alloc : memref<4x4xi32, 1>
        }
        %async_token_6, %results_7 = air.execute -> (memref<4x4xi32, 1>) {
          %alloc = memref.alloc() : memref<4x4xi32, 1>
          air.execute_terminator %alloc : memref<4x4xi32, 1>
        }
        %4 = air.channel.get async [%async_token_4]  @channel_0[] (%results_5[] [] []) {id = 3 : i32} : (memref<4x4xi32, 1>)
        %5 = air.channel.get async [%async_token_6]  @channel_1[] (%results_7[] [] []) {id = 4 : i32} : (memref<4x4xi32, 1>)
        %6 = air.herd @herd_0 async  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) attributes {id = 3 : i32} 
        %async_token_8 = air.execute [%4] {
          memref.dealloc %results_5 : memref<4x4xi32, 1>
        }
        %async_token_9 = air.execute [%5] {
          memref.dealloc %results_7 : memref<4x4xi32, 1>
        }
      }
      %async_token_2 = air.execute [%1] {
        memref.dealloc %results : memref<4x4xi32>
      }
      %async_token_3 = air.execute [%2] {
        memref.dealloc %results_1 : memref<4x4xi32>
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func3
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.channel.get{{.*}}@channel_1
// AGGRESSIVE-LABEL: func3
// AGGRESSIVE: air.launch
// AGGRESSIVE: air.channel.put{{.*}}@channel_1
// AGGRESSIVE: air.channel.put{{.*}}@channel_1
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: air.channel.get{{.*}}@channel_1
// AGGRESSIVE: air.channel.get{{.*}}@channel_1
// AGGL1-LABEL: func3
// AGGL1: air.launch
// AGGL1: air.channel.put{{.*}}@channel_1
// AGGL1: air.channel.put{{.*}}@channel_1
// AGGL1: air.segment @segment_0
// AGGL1: air.herd @herd_0
// AGGL1: air.channel.get{{.*}}@channel_1
// AGGL1: air.channel.get{{.*}}@channel_1

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_1 [2, 2]
  air.channel @channel_0 [2, 2]
  func.func @func3(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1) : memref<64x64xi32>, memref<64x64xi32> {
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %0 = air.wait_all async 
      %1 = scf.parallel (%arg9, %arg10) = (%c0, %c0) to (%c2, %c2) step (%c1_0, %c1_0) init (%0) -> !air.async.token {
        %4 = affine.apply #map()[%arg9]
        %5 = arith.addi %arg3, %4 : index
        scf.for %arg11 = %c0 to %c64 step %c32 {
          air.channel.put  @channel_0[%arg9, %arg10] (%arg7[%5, %arg11] [%c32, %c32] [%c64, %c1_0]) : (memref<64x64xi32>)
        }
        %6 = air.wait_all async 
        scf.reduce(%6 : !air.async.token) {
        ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
          %7 = air.wait_all async [%arg11, %arg12] 
          scf.reduce.return %7 : !air.async.token
        }
      }
      %2 = air.wait_all async 
      %3 = scf.parallel (%arg9, %arg10) = (%c0, %c0) to (%c2, %c2) step (%c1_0, %c1_0) init (%2) -> !air.async.token {
        %4 = affine.apply #map()[%arg10]
        %5 = arith.addi %arg4, %4 : index
        scf.for %arg11 = %c0 to %c64 step %c32 {
          air.channel.put  @channel_1[%arg9, %arg10] (%arg8[%arg11, %5] [%c32, %c32] [%c64, %c1_0]) : (memref<64x64xi32>)
        }
        %6 = air.wait_all async 
        scf.reduce(%6 : !air.async.token) {
        ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
          %7 = air.wait_all async [%arg11, %arg12] 
          scf.reduce.return %7 : !air.async.token
        }
      }
      air.segment @segment_0  {
        %c2_1 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg9, %arg10) in (%arg11=%c2_1, %arg12=%c2_1) {
          %c0_2 = arith.constant 0 : index
          %c64_3 = arith.constant 64 : index
          %c32_4 = arith.constant 32 : index
          scf.for %arg13 = %c0_2 to %c64_3 step %c32_4 {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            %alloc_5 = memref.alloc() : memref<32x32xi32, 2>
            air.channel.get  @channel_0[%arg9, %arg10] (%alloc[] [] []) : (memref<32x32xi32, 2>)
            air.channel.get  @channel_1[%arg9, %arg10] (%alloc_5[] [] []) : (memref<32x32xi32, 2>)
            memref.dealloc %alloc : memref<32x32xi32, 2>
            memref.dealloc %alloc_5 : memref<32x32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func4
// CHECK: air.launch
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_2
// CHECK: air.channel.get{{.*}}@channel_3
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK: air.channel.put{{.*}}@channel_2
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK: air.channel.put{{.*}}@channel_3
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_2
// CHECK: air.channel.get{{.*}}@channel_3
// AGGRESSIVE-LABEL: func4
// AGGRESSIVE: air.launch
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: air.channel.get{{.*}}@channel_2
// AGGRESSIVE: air.channel.get{{.*}}@channel_3
// AGGRESSIVE: scf.for
// AGGRESSIVE-NEXT: scf.for
// AGGRESSIVE: air.channel.put{{.*}}@channel_2
// AGGRESSIVE: scf.for
// AGGRESSIVE-NEXT: scf.for
// AGGRESSIVE: air.channel.put{{.*}}@channel_3
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: air.channel.get{{.*}}@channel_2
// AGGRESSIVE: air.channel.get{{.*}}@channel_3
// AGGL1-LABEL: func4
// AGGL1: air.launch
// AGGL1: air.segment @segment_0
// AGGL1: air.herd @herd_0
// AGGL1: air.channel.get{{.*}}@channel_2
// AGGL1: air.channel.get{{.*}}@channel_3
// AGGL1: scf.for
// AGGL1-NEXT: scf.for
// AGGL1: air.channel.put{{.*}}@channel_2
// AGGL1: scf.for
// AGGL1-NEXT: scf.for
// AGGL1: air.channel.put{{.*}}@channel_3
// AGGL1: air.herd @herd_0
// AGGL1: air.channel.get{{.*}}@channel_2
// AGGL1: air.channel.get{{.*}}@channel_3

#map = affine_map<(d0) -> (d0 * 8)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func @func4(%arg0: memref<1024x2048xi32>, %arg1: memref<2048x512xi32>, %arg2: memref<1024x512xi32>) {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %async_token, %results = air.execute -> (memref<1x1x4x8x4x8xi32, 2>) {
      %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
      air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2>
    }
    %async_token_0, %results_1 = air.execute -> (memref<2x1x32x256xi32, 1>) {
      %alloc = memref.alloc() : memref<2x1x32x256xi32, 1>
      air.execute_terminator %alloc : memref<2x1x32x256xi32, 1>
    }
    %0 = air.launch async [%async_token, %async_token_0] (%arg3, %arg4) in (%arg5=%c16, %arg6=%c8) args(%arg7=%results_1, %arg8=%results) : memref<2x1x32x256xi32, 1>, memref<1x1x4x8x4x8xi32, 2> attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  args(%arg9=%arg7, %arg10=%arg8) : memref<2x1x32x256xi32, 1>, memref<1x1x4x8x4x8xi32, 2> attributes {id = 2 : i32} {
        %c8192 = arith.constant 8192 : index
        %c4 = arith.constant 4 : index
        %c1024 = arith.constant 1024 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c8_4 = arith.constant 8 : index
        %2 = air.wait_all async 
        %3 = scf.for %arg11 = %c0 to %c32 step %c4 iter_args(%arg12 = %2) -> (!air.async.token) {
          %async_token_5, %results_6 = air.execute [%arg12] -> (index) {
            %13 = affine.apply #map(%arg11)
            air.execute_terminator %13 : index
          }
          %12 = air.channel.put async [%async_token_5]  @channel_0[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 9 : i32} : (memref<2x1x32x256xi32, 1>)
          scf.yield %12 : !air.async.token
        }
        %4 = air.wait_all async 
        %5 = scf.for %arg11 = %c0 to %c32 step %c4 iter_args(%arg12 = %4) -> (!air.async.token) {
          %async_token_5, %results_6 = air.execute [%arg12] -> (index) {
            %13 = affine.apply #map(%arg11)
            air.execute_terminator %13 : index
          }
          %12 = air.channel.put async [%async_token_5]  @channel_1[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 10 : i32} : (memref<2x1x32x256xi32, 1>)
          scf.yield %12 : !air.async.token
        }
        %6 = air.herd @herd_0 async  tile (%arg11, %arg12) in (%arg13=%c2, %arg14=%c2) args(%arg15=%arg10) : memref<1x1x4x8x4x8xi32, 2> attributes {id = 3 : i32} {
          %c0_5 = arith.constant 0 : index
          %c32_6 = arith.constant 32 : index
          %c4_7 = arith.constant 4 : index
          %12 = air.wait_all async 
          %13 = scf.for %arg16 = %c0_5 to %c32_6 step %c4_7 iter_args(%arg17 = %12) -> (!air.async.token) {
            %14 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
              %15 = air.channel.get async [%arg17]  @channel_0[%arg11, %arg12] (%arg15[] [] []) {id = 13 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
              affine.yield %15 : !air.async.token
            } else {
              %15 = air.channel.get async [%arg17]  @channel_1[%arg11, %arg12] (%arg15[] [] []) {id = 14 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
              affine.yield %15 : !air.async.token
            }
            scf.yield %14 : !air.async.token
          }
        }
        %7 = air.wait_all async 
        %8 = scf.for %arg11 = %c1 to %c8_4 step %c1 iter_args(%arg12 = %7) -> (!air.async.token) {
          %12 = scf.for %arg13 = %c0 to %c32 step %c4 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_5, %results_6 = air.execute [%arg14] -> (index) {
              %14 = affine.apply #map(%arg13)
              air.execute_terminator %14 : index
            }
            %13 = air.channel.put async [%async_token_5]  @channel_2[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 19 : i32} : (memref<2x1x32x256xi32, 1>)
            scf.yield %13 : !air.async.token
          }
          scf.yield %12 : !air.async.token
        }
        %9 = air.wait_all async 
        %10 = scf.for %arg11 = %c1 to %c8_4 step %c1 iter_args(%arg12 = %9) -> (!air.async.token) {
          %12 = scf.for %arg13 = %c0 to %c32 step %c4 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_5, %results_6 = air.execute [%arg14] -> (index) {
              %14 = affine.apply #map(%arg13)
              air.execute_terminator %14 : index
            }
            %13 = air.channel.put async [%async_token_5]  @channel_3[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 20 : i32} : (memref<2x1x32x256xi32, 1>)
            scf.yield %13 : !air.async.token
          }
          scf.yield %12 : !air.async.token
        }
        %11 = air.herd @herd_0 async  tile (%arg11, %arg12) in (%arg13=%c2, %arg14=%c2) args(%arg15=%arg10) : memref<1x1x4x8x4x8xi32, 2> attributes {id = 4 : i32} {
          %c4_5 = arith.constant 4 : index
          %c32_6 = arith.constant 32 : index
          %c0_7 = arith.constant 0 : index
          %c1_8 = arith.constant 1 : index
          %c8_9 = arith.constant 8 : index
          scf.for %arg16 = %c1_8 to %c8_9 step %c1_8 {
            %12 = air.wait_all async 
            %13 = scf.for %arg17 = %c0_7 to %c32_6 step %c4_5 iter_args(%arg18 = %12) -> (!air.async.token) {
              %14 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
                %15 = air.channel.get async [%arg18]  @channel_2[%arg11, %arg12] (%arg15[] [] []) {id = 23 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
                affine.yield %15 : !air.async.token
              } else {
                %15 = air.channel.get async [%arg18]  @channel_3[%arg11, %arg12] (%arg15[] [] []) {id = 24 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
                affine.yield %15 : !air.async.token
              }
              scf.yield %14 : !air.async.token
            }
          }
        }
      }
    }
    %async_token_2 = air.execute [%0] {
      memref.dealloc %results_1 : memref<2x1x32x256xi32, 1>
    }
    %async_token_3 = air.execute [%0] {
      memref.dealloc %results : memref<1x1x4x8x4x8xi32, 2>
    }
    return
  }
}

// -----

// Merging air.channels into both scf.for op's LB and UB (L3->L2).

// CHECK-LABEL: func5
// CHECK: air.launch
// CHECK: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// CHECK: air.channel.put{{.*}}@channel_4
// CHECK: scf.yield
// CHECK: air.segment @segment_0
// CHECK: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// CHECK: air.channel.get{{.*}}@channel_4
// CHECK: scf.yield
// AGGRESSIVE-LABEL: func5
// AGGRESSIVE: air.launch
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// AGGRESSIVE: air.channel.put{{.*}}@channel_4
// AGGRESSIVE: scf.yield
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// AGGRESSIVE: air.channel.get{{.*}}@channel_4
// AGGRESSIVE: scf.yield
// AGGL1-LABEL: func5
// AGGL1: air.launch
// AGGL1: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// AGGL1: air.channel.put{{.*}}@channel_4
// AGGL1: scf.yield
// AGGL1: air.segment @segment_0
// AGGL1: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// AGGL1: air.channel.get{{.*}}@channel_4
// AGGL1: scf.yield

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_8 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @func5(%arg0: memref<1024x512xi8>) {
    %c16 = arith.constant 16 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c16, %arg7=%c16) args(%arg8=%arg0) : memref<1024x512xi8> attributes {id = 1 : i32} {
      %c480 = arith.constant 480 : index
      %c15 = arith.constant 15 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %6 = affine.apply #map()[%arg4]
        air.execute_terminator %6 : index
      }
      %1 = air.channel.put async [%async_token]  @channel_0[] (%arg8[%results, %c0] [%c64, %c32] [%c512, %c1]) {id = 1 : i32} : (memref<1024x512xi8>)
      %async_token_0, %results_1 = air.execute -> (index) {
        %6 = affine.apply #map()[%arg4]
        air.execute_terminator %6 : index
      }
      %2 = air.wait_all async 
      %3 = scf.for %arg9 = %c1 to %c15 step %c1 iter_args(%arg10 = %2) -> (!air.async.token) {
        %async_token_4, %results_5 = air.execute [%arg10] -> (index) {
          %7 = affine.apply #map1()[%arg9]
          air.execute_terminator %7 : index
        }
        %6 = air.channel.put async [%async_token_4, %async_token_0]  @channel_4[] (%arg8[%results_1, %results_5] [%c64, %c32] [%c512, %c1]) {id = 3 : i32} : (memref<1024x512xi8>)
        scf.yield %6 : !air.async.token
      }
      %async_token_2, %results_3 = air.execute -> (index) {
        %6 = affine.apply #map()[%arg4]
        air.execute_terminator %6 : index
      }
      %4 = air.channel.put async [%async_token_2]  @channel_8[] (%arg8[%results_3, %c480] [%c64, %c32] [%c512, %c1]) {id = 5 : i32} : (memref<1024x512xi8>)
      %5 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c1_4 = arith.constant 1 : index
        %c15_5 = arith.constant 15 : index
        %6 = air.wait_all async 
        %async_token_6, %results_7 = air.execute -> (memref<1x1x64x32xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x32xi8, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x32xi8, 1 : i32>
        }
        %7 = air.channel.get async [%6, %async_token_6]  @channel_0[] (%results_7[] [] []) {id = 10 : i32} : (memref<1x1x64x32xi8, 1 : i32>)
        %8 = air.wait_all async [%async_token_6, %7] 
        %9 = scf.for %arg9 = %c1_4 to %c15_5 step %c1_4 iter_args(%arg10 = %8) -> (!air.async.token) {
          %11 = air.channel.get async [%arg10]  @channel_4[] (%results_7[] [] []) {id = 16 : i32} : (memref<1x1x64x32xi8, 1 : i32>)
          scf.yield %11 : !air.async.token
        }
        %10 = air.channel.get async [%9]  @channel_8[] (%results_7[] [] []) {id = 22 : i32} : (memref<1x1x64x32xi8, 1 : i32>)
        %async_token_8 = air.execute [%10] {
          memref.dealloc %results_7 : memref<1x1x64x32xi8, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Merging air.channels into both scf.for op's LB and UB (L2->L1).

// CHECK-LABEL: func6
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_6
// CHECK: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// CHECK-NEXT: scf.parallel
// CHECK: air.channel.put{{.*}}@channel_6
// CHECK: scf.reduce
// CHECK: scf.yield
// CHECK: air.herd @herd_0
// CHECK: scf.for %{{.*}} = %c1{{.*}}to %c15{{.*}}step %c1
// CHECK: air.channel.get{{.*}}@channel_6
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_6
// AGGRESSIVE-LABEL: func6
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: air.channel.get{{.*}}@channel_6
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// AGGRESSIVE-NEXT: scf.parallel
// AGGRESSIVE: air.channel.put{{.*}}@channel_6
// AGGRESSIVE: scf.reduce
// AGGRESSIVE: scf.yield
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: scf.for %{{.*}} = %c1{{.*}}to %c15{{.*}}step %c1
// AGGRESSIVE: air.channel.get{{.*}}@channel_6
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: air.channel.get{{.*}}@channel_6
// AGGL1-LABEL: func6
// AGGL1: air.segment @segment_0
// AGGL1: air.herd @herd_0
// AGGL1: air.channel.get{{.*}}@channel_6
// AGGL1: scf.for %{{.*}} = %c0{{.*}}to %c16{{.*}}step %c1{{.*}}iter_args
// AGGL1-NEXT: scf.parallel
// AGGL1: air.channel.put{{.*}}@channel_6
// AGGL1: scf.reduce
// AGGL1: scf.yield
// AGGL1: air.herd @herd_0
// AGGL1: scf.for %{{.*}} = %c1{{.*}}to %c15{{.*}}step %c1
// AGGL1: air.channel.get{{.*}}@channel_6
// AGGL1: air.herd @herd_0
// AGGL1: air.channel.get{{.*}}@channel_6

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_12 [2, 2]
  air.channel @channel_6 [2, 2]
  air.channel @channel_2 [2, 2]
  func.func @func6() {
    %0 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2048 = arith.constant 2048 : index
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c15 = arith.constant 15 : index
      %async_token_6, %results_7 = air.execute -> (memref<1x1x4x8x4x8xi8, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x4x8x4x8xi8, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x4x8x4x8xi8, 2 : i32>
      }
      %async_token_8, %results_9 = air.execute -> (memref<1x1x64x32xi8, 1 : i32>) {
        %alloc = memref.alloc() : memref<1x1x64x32xi8, 1 : i32>
        air.execute_terminator %alloc : memref<1x1x64x32xi8, 1 : i32>
      }
      %1 = scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_8) -> !air.async.token {
        %async_token_12, %results_13 = air.execute -> (index) {
          %8 = affine.apply #map()[%arg0]
          air.execute_terminator %8 : index
        }
        %7 = air.channel.put async [%async_token_8, %async_token_12]  @channel_2[%arg0, %arg1] (%results_9[%c0, %c0, %c0, %c0, %results_13, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1]) {id = 12 : i32} : (memref<1x1x64x32xi8, 1 : i32>)
        scf.reduce(%7 : !air.async.token) {
        ^bb0(%arg2: !air.async.token, %arg3: !air.async.token):
          %8 = air.wait_all async [%arg2, %arg3] 
          scf.reduce.return %8 : !air.async.token
        }
      }
      %2 = air.herd @herd_0 async [%async_token_6, %async_token_8]  tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c2) args(%arg4=%results_7) : memref<1x1x4x8x4x8xi8, 2 : i32> attributes {id = 3 : i32} {
        %7 = air.wait_all async 
        %8 = air.channel.get async [%7]  @channel_2[%arg0, %arg1] (%arg4[] [] []) {id = 14 : i32} : (memref<1x1x4x8x4x8xi8, 2 : i32>)
      }
      %3 = scf.for %arg0 = %c1 to %c15 step %c1 iter_args(%arg1 = %async_token_8) -> (!air.async.token) {
        %7 = scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%arg1) -> !air.async.token {
          %async_token_12, %results_13 = air.execute -> (index) {
            %9 = affine.apply #map()[%arg2]
            air.execute_terminator %9 : index
          }
          %8 = air.channel.put async [%arg1, %async_token_12]  @channel_6[%arg2, %arg3] (%results_9[%c0, %c0, %c0, %c0, %results_13, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1]) {id = 18 : i32} : (memref<1x1x64x32xi8, 1 : i32>)
          scf.reduce(%8 : !air.async.token) {
          ^bb0(%arg4: !air.async.token, %arg5: !air.async.token):
            %9 = air.wait_all async [%arg4, %arg5] 
            scf.reduce.return %9 : !air.async.token
          }
        }
        scf.yield %7 : !air.async.token
      }
      %4 = air.herd @herd_0 async [%2]  tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c2) args(%arg4=%results_7) : memref<1x1x4x8x4x8xi8, 2 : i32> attributes {id = 4 : i32} {
        %c1_12 = arith.constant 1 : index
        %c15_13 = arith.constant 15 : index
        scf.for %arg5 = %c1_12 to %c15_13 step %c1_12 {
          %7 = air.wait_all async 
          %8 = air.channel.get async [%7]  @channel_6[%arg0, %arg1] (%arg4[] [] []) {id = 20 : i32} : (memref<1x1x4x8x4x8xi8, 2 : i32>)
        }
      }
      %5 = scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%2) -> !air.async.token {
        %async_token_12, %results_13 = air.execute -> (index) {
          %8 = affine.apply #map()[%arg0]
          air.execute_terminator %8 : index
        }
        %7 = air.channel.put async [%2, %async_token_12]  @channel_12[%arg0, %arg1] (%results_9[%c0, %c0, %c0, %c0, %results_13, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1]) {id = 26 : i32} : (memref<1x1x64x32xi8, 1 : i32>)
        scf.reduce(%7 : !air.async.token) {
        ^bb0(%arg2: !air.async.token, %arg3: !air.async.token):
          %8 = air.wait_all async [%arg2, %arg3] 
          scf.reduce.return %8 : !air.async.token
        }
      }
      %6 = air.herd @herd_0 async [%4]  tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c2) args(%arg4=%results_7) : memref<1x1x4x8x4x8xi8, 2 : i32> attributes {id = 5 : i32} {
        %7 = air.wait_all async 
        %8 = air.channel.get async [%7]  @channel_12[%arg0, %arg1] (%arg4[] [] []) {id = 30 : i32} : (memref<1x1x4x8x4x8xi8, 2 : i32>)
      }
      %async_token_10 = air.execute {
        memref.dealloc %results_9 : memref<1x1x64x32xi8, 1 : i32>
      }
      %async_token_11 = air.execute [%6] {
        memref.dealloc %results_7 : memref<1x1x4x8x4x8xi8, 2 : i32>
      }
    }
    return
  }
}

// -----

// Merging air.channels into both scf.for op's LB and UB (L2->L1, with broadcast).

// CHECK-LABEL: func7
// CHECK: air.segment @segment_0
// CHECK: scf.for %{{.*}} = %c0{{.*}}to %c64{{.*}}step %c1{{.*}}{
// CHECK-NEXT: air.channel.put{{.*}}@channel_6{{.*}} : (memref<1x1x32x64xi32, 1 : i32>)
// CHECK-NEXT: }
// CHECK: scf.for %{{.*}} = %c0{{.*}}to %c64{{.*}}step %c1{{.*}}{
// CHECK-NEXT: air.channel.put{{.*}}@channel_7{{.*}} : (memref<1x1x32x64xi32, 1 : i32>)
// CHECK-NEXT: }
// AGGRESSIVE-LABEL: func7
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}}to %c64{{.*}}step %c1{{.*}}{
// AGGRESSIVE-NEXT: air.channel.put{{.*}}@channel_6{{.*}} : (memref<1x1x32x64xi32, 1 : i32>)
// AGGRESSIVE-NEXT: }
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}}to %c64{{.*}}step %c1{{.*}}{
// AGGRESSIVE-NEXT: air.channel.put{{.*}}@channel_7{{.*}} : (memref<1x1x32x64xi32, 1 : i32>)
// AGGRESSIVE-NEXT: }
// AGGL1-LABEL: func7
// AGGL1: air.segment @segment_0
// AGGL1: scf.for %{{.*}} = %c0{{.*}}to %c64{{.*}}step %c1{{.*}}{
// AGGL1-NEXT: air.channel.put{{.*}}@channel_6{{.*}} : (memref<1x1x32x64xi32, 1 : i32>)
// AGGL1-NEXT: }
// AGGL1: scf.for %{{.*}} = %c0{{.*}}to %c64{{.*}}step %c1{{.*}}{
// AGGL1-NEXT: air.channel.put{{.*}}@channel_7{{.*}} : (memref<1x1x32x64xi32, 1 : i32>)
// AGGL1-NEXT: }

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
module {
  air.channel @channel_11 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_10 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_7 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_6 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]}
  func.func @func7(%arg0: memref<2048x2048xi32>, %arg1: memref<2048x2048xi32>, %arg2: memref<2048x2048xi32>) {
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c512 = arith.constant 512 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c32_0 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c2048 = arith.constant 2048 : index
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c63 = arith.constant 63 : index
        %async_token, %results = air.execute -> (memref<1x1x8x4x8x4xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
        }
        %async_token_1, %results_2 = air.execute -> (memref<1x1x32x64xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x32x64xi32, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x32x64xi32, 1 : i32>
        }
        %2 = air.channel.put async [%async_token_1]  @channel_2[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512, %c64, %c1]) {id = 12 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        %3 = air.channel.put async [%async_token_1]  @channel_3[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c32_0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512, %c64, %c1]) {id = 13 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        %4 = air.herd @herd_0 async [%async_token_1]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x8x4x8x4xi32, 2 : i32> attributes {id = 3 : i32} {
          %9 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
            %10 = air.channel.get async  @channel_2[%arg7, %arg8] (%arg11[] [] []) {id = 16 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
            affine.yield %10 : !air.async.token
          } else {
            %10 = air.channel.get async  @channel_3[%arg7, %arg8] (%arg11[] [] []) {id = 17 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
            affine.yield %10 : !air.async.token
          }
        }
        scf.for %arg7 = %c1 to %c63 step %c1 {
          %9 = air.channel.put async [%async_token_1]  @channel_6[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512, %c64, %c1]) {id = 22 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c63 step %c1 {
          %9 = air.channel.put async [%async_token_1]  @channel_7[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c32_0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512, %c64, %c1]) {id = 23 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        }
        %5 = air.herd @herd_0 async [%4]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x8x4x8x4xi32, 2 : i32> attributes {id = 4 : i32} {
          %c1_4 = arith.constant 1 : index
          %c63_5 = arith.constant 63 : index
          scf.for %arg12 = %c1_4 to %c63_5 step %c1_4 {
            %9 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
              %10 = air.channel.get async  @channel_6[%arg7, %arg8] (%arg11[] [] []) {id = 26 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
              affine.yield %10 : !air.async.token
            } else {
              %10 = air.channel.get async  @channel_7[%arg7, %arg8] (%arg11[] [] []) {id = 27 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
              affine.yield %10 : !air.async.token
            }
          }
        }
        %6 = air.channel.put async [%5]  @channel_10[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512, %c64, %c1]) {id = 32 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        %7 = air.channel.put async [%5]  @channel_11[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c32_0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512, %c64, %c1]) {id = 33 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        %8 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x8x4x8x4xi32, 2 : i32> attributes {id = 5 : i32} {
          %9 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
            %10 = air.channel.get async  @channel_10[%arg7, %arg8] (%arg11[] [] []) {id = 37 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
            affine.yield %10 : !air.async.token
          } else {
            %10 = air.channel.get async  @channel_11[%arg7, %arg8] (%arg11[] [] []) {id = 38 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
            affine.yield %10 : !air.async.token
          }
        }
        %async_token_3 = air.execute [%8] {
          memref.dealloc %results_2 : memref<1x1x32x64xi32, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Fusing air.channels when there are no scf.for loops (NFL) to merge into.

// CHECK-LABEL: func8
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: affine.if
// CHECK: air.channel.get{{.*}}@channel_2{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// CHECK: affine.yield
// CHECK: else
// CHECK: air.channel.get{{.*}}@channel_3{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// CHECK: affine.yield
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_2{{.*}} : (memref<1x1x64x32xi32, 1 : i32>)
// CHECK-NEXT: scf.yield
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_3{{.*}} : (memref<1x1x64x32xi32, 1 : i32>)
// CHECK-NEXT: scf.yield
// CHECK: air.herd @herd_0
// CHECK: affine.if
// CHECK: air.channel.get{{.*}}@channel_2{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// CHECK: affine.yield
// CHECK: else
// CHECK: air.channel.get{{.*}}@channel_3{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// CHECK: affine.yield
// AGGRESSIVE-LABEL: func8
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: affine.if
// AGGRESSIVE: air.channel.get{{.*}}@channel_2{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGRESSIVE: affine.yield
// AGGRESSIVE: else
// AGGRESSIVE: air.channel.get{{.*}}@channel_3{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGRESSIVE: affine.yield
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: air.channel.put{{.*}}@channel_2{{.*}} : (memref<1x1x64x32xi32, 1 : i32>)
// AGGRESSIVE-NEXT: scf.yield
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: air.channel.put{{.*}}@channel_3{{.*}} : (memref<1x1x64x32xi32, 1 : i32>)
// AGGRESSIVE-NEXT: scf.yield
// AGGRESSIVE: air.herd @herd_0
// AGGRESSIVE: affine.if
// AGGRESSIVE: air.channel.get{{.*}}@channel_2{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGRESSIVE: affine.yield
// AGGRESSIVE: else
// AGGRESSIVE: air.channel.get{{.*}}@channel_3{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGRESSIVE: affine.yield
// AGGL1-LABEL: func8
// AGGL1: air.segment @segment_0
// AGGL1: air.herd @herd_0
// AGGL1: affine.if
// AGGL1: air.channel.get{{.*}}@channel_2{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGL1: affine.yield
// AGGL1: else
// AGGL1: air.channel.get{{.*}}@channel_3{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGL1: affine.yield
// AGGL1: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: air.channel.put{{.*}}@channel_2{{.*}} : (memref<1x1x64x32xi32, 1 : i32>)
// AGGL1-NEXT: scf.yield
// AGGL1: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: air.channel.put{{.*}}@channel_3{{.*}} : (memref<1x1x64x32xi32, 1 : i32>)
// AGGL1-NEXT: scf.yield
// AGGL1: air.herd @herd_0
// AGGL1: affine.if
// AGGL1: air.channel.get{{.*}}@channel_2{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGL1: affine.yield
// AGGL1: else
// AGGL1: air.channel.get{{.*}}@channel_3{{.*}} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
// AGGL1: affine.yield

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func @func8(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c2048 = arith.constant 2048 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<1x1x4x8x4x8xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
        }
        %async_token_1, %results_2 = air.execute -> (memref<1x1x64x32xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x32xi32, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x32xi32, 1 : i32>
        }
        %2 = air.channel.put async [%async_token_1]  @channel_0[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c0] [%c1_0, %c1_0, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1_0]) {id = 8 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        %3 = air.channel.put async [%async_token_1]  @channel_1[] (%results_2[%c0, %c0, %c0, %c0, %c32, %c0] [%c1_0, %c1_0, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1_0]) {id = 9 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        %4 = air.herd @herd_0 async [%async_token, %async_token_1]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x4x8x4x8xi32, 2 : i32> attributes {id = 3 : i32} {
          %8 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
            %9 = air.channel.get async  @channel_0[%arg7, %arg8] (%arg11[] [] []) {id = 12 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
            affine.yield %9 : !air.async.token
          } else {
            %9 = air.channel.get async  @channel_1[%arg7, %arg8] (%arg11[] [] []) {id = 13 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
            affine.yield %9 : !air.async.token
          }
        }
        %5 = air.channel.put async [%4]  @channel_2[] (%results_2[%c0, %c0, %c0, %c0, %c0, %c0] [%c1_0, %c1_0, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1_0]) {id = 18 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        %6 = air.channel.put async [%4]  @channel_3[] (%results_2[%c0, %c0, %c0, %c0, %c32, %c0] [%c1_0, %c1_0, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32, %c1_0]) {id = 19 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        %7 = air.herd @herd_0 async [%4]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x4x8x4x8xi32, 2 : i32> attributes {id = 4 : i32} {
          %8 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
            %9 = air.channel.get async  @channel_2[%arg7, %arg8] (%arg11[] [] []) {id = 23 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
            affine.yield %9 : !air.async.token
          } else {
            %9 = air.channel.get async  @channel_3[%arg7, %arg8] (%arg11[] [] []) {id = 24 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
            affine.yield %9 : !air.async.token
          }
        }
        %async_token_3 = air.execute [%7] {
          memref.dealloc %results_2 : memref<1x1x64x32xi32, 1 : i32>
        }
        %async_token_4 = air.execute [%7] {
          memref.dealloc %results : memref<1x1x4x8x4x8xi32, 2 : i32>
        }
      }
    }
    return
  }
}

// -----

// No-for-loop (NFL) mode: avoid fusing into a new for loop if differing data access pattern.
// Note: see L3-side channel puts in example below not being fused into a new for loop.

// CHECK-LABEL: func9
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_4{{.*}} : (memref<512x256xi8>)
// CHECK: air.channel.put{{.*}}@channel_5{{.*}} : (memref<256x32xi8>)
// CHECK: air.channel.put{{.*}}@channel_4{{.*}} : (memref<512x256xi8>)
// CHECK: air.channel.put{{.*}}@channel_5{{.*}} : (memref<256x32xi8>)
// CHECK: air.segment @segment_0
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.get{{.*}}@channel_4{{.*}} : (memref<2x1x128x128xi8, 1 : i32>)
// CHECK-NEXT: scf.yield
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.get{{.*}}@channel_5{{.*}} : (memref<1x2x128x16xi8, 1 : i32>)
// CHECK-NEXT: scf.yield
// AGGRESSIVE: air.launch
// AGGRESSIVE: air.channel.put{{.*}}@channel_5{{.*}} : (memref<512x256xi8>)
// AGGRESSIVE: air.channel.put{{.*}}@channel_5{{.*}} : (memref<256x32xi8>)
// AGGRESSIVE: air.channel.put{{.*}}@channel_5{{.*}} : (memref<512x256xi8>)
// AGGRESSIVE: air.channel.put{{.*}}@channel_5{{.*}} : (memref<256x32xi8>)
// AGGRESSIVE: air.segment @segment_0
// AGGRESSIVE: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: %[[TOK0:.*]] = air.channel.get{{.*}}@channel_5{{.*}} : (memref<1x2x128x16xi8, 1 : i32>)
// AGGRESSIVE-NEXT: %[[TOK1:.*]] = air.channel.get{{.*}}@channel_5{{.*}} : (memref<2x1x128x128xi8, 1 : i32>)
// AGGRESSIVE-NEXT: air.wait_all async [%[[TOK0]], %[[TOK1]]] 
// AGGRESSIVE-NEXT: scf.yield
// AGGL1: air.launch
// AGGL1: air.channel.put{{.*}}@channel_4{{.*}} : (memref<512x256xi8>)
// AGGL1: air.channel.put{{.*}}@channel_5{{.*}} : (memref<256x32xi8>)
// AGGL1: air.channel.put{{.*}}@channel_4{{.*}} : (memref<512x256xi8>)
// AGGL1: air.channel.put{{.*}}@channel_5{{.*}} : (memref<256x32xi8>)
// AGGL1: air.segment @segment_0
// AGGL1: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: air.channel.get{{.*}}@channel_4{{.*}} : (memref<2x1x128x128xi8, 1 : i32>)
// AGGL1-NEXT: scf.yield
// AGGL1: scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: air.channel.get{{.*}}@channel_5{{.*}} : (memref<1x2x128x16xi8, 1 : i32>)
// AGGL1-NEXT: scf.yield

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @func9(%arg0: memref<512x256xi8>, %arg1: memref<256x32xi8>, %arg2: memref<512x32xi32>, %arg3: memref<512x32xi32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<512x256xi8>, memref<256x32xi8> attributes {id = 1 : i32} {
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1_0 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c2_1 = arith.constant 2 : index
      %async_token, %results = air.execute -> (index) {
        %6 = affine.apply #map()[%arg4]
        air.execute_terminator %6 : index
      }
      %1 = air.channel.put async [%async_token]  @channel_0[] (%arg8[%c0, %c0, %results, %c0] [%c2_1, %c1_0, %c128, %c128] [%c32768, %c128, %c256, %c1_0]) {id = 1 : i32} : (memref<512x256xi8>)
      %async_token_2, %results_3 = air.execute -> (index) {
        %6 = affine.apply #map1()[%arg5]
        air.execute_terminator %6 : index
      }
      %2 = air.channel.put async [%async_token_2]  @channel_1[] (%arg9[%c0, %c0, %c0, %results_3] [%c1_0, %c2_1, %c128, %c16] [%c4096, %c16, %c32, %c1_0]) {id = 2 : i32} : (memref<256x32xi8>)
      %async_token_4, %results_5 = air.execute -> (index) {
        %6 = affine.apply #map()[%arg4]
        air.execute_terminator %6 : index
      }
      %3 = air.channel.put async [%async_token_4]  @channel_4[] (%arg8[%c0, %c0, %results_5, %c128] [%c2_1, %c1_0, %c128, %c128] [%c32768, %c128, %c256, %c1_0]) {id = 3 : i32} : (memref<512x256xi8>)
      %async_token_6, %results_7 = air.execute -> (index) {
        %6 = affine.apply #map1()[%arg5]
        air.execute_terminator %6 : index
      }
      %4 = air.channel.put async [%async_token_6]  @channel_5[] (%arg9[%c0, %c0, %c128, %results_7] [%c1_0, %c2_1, %c128, %c16] [%c4096, %c16, %c32, %c1_0]) {id = 4 : i32} : (memref<256x32xi8>)
      %5 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %6 = air.wait_all async 
        %7 = air.wait_all async 
        %async_token_8, %results_9 = air.execute -> (memref<1x2x128x16xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x2x128x16xi8, 1 : i32>
          air.execute_terminator %alloc : memref<1x2x128x16xi8, 1 : i32>
        }
        %async_token_10, %results_11 = air.execute -> (memref<2x1x128x128xi8, 1 : i32>) {
          %alloc = memref.alloc() : memref<2x1x128x128xi8, 1 : i32>
          air.execute_terminator %alloc : memref<2x1x128x128xi8, 1 : i32>
        }
        %8 = air.channel.get async [%7, %async_token_10]  @channel_0[] (%results_11[] [] []) {id = 7 : i32} : (memref<2x1x128x128xi8, 1 : i32>)
        %9 = air.channel.get async [%6, %async_token_8]  @channel_1[] (%results_9[] [] []) {id = 8 : i32} : (memref<1x2x128x16xi8, 1 : i32>)
        %10 = air.channel.get async [%8]  @channel_4[] (%results_11[] [] []) {id = 13 : i32} : (memref<2x1x128x128xi8, 1 : i32>)
        %11 = air.channel.get async [%9]  @channel_5[] (%results_9[] [] []) {id = 14 : i32} : (memref<1x2x128x16xi8, 1 : i32>)
        %async_token_12 = air.execute [%10] {
          memref.dealloc %results_11 : memref<2x1x128x128xi8, 1 : i32>
        }
        %async_token_13 = air.execute [%11] {
          memref.dealloc %results_9 : memref<1x2x128x16xi8, 1 : i32>
        }
      }
    }
    return
  }
}

// -----

// Channel fusion under multiple herds and multiple affine.if conditions.

// CHECK-LABEL: func10
// CHECK: air.herd @herd_0
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_8{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_9{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_10{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: else
// CHECK-NEXT: air.channel.get{{.*}}@channel_11{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_12{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_13{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_14{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: else
// CHECK-NEXT: air.channel.get{{.*}}@channel_15{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_8{{.*}} : (memref<1x1x256x64xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_9{{.*}} : (memref<1x1x256x64xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_10{{.*}} : (memref<1x1x256x64xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_11{{.*}} : (memref<1x1x256x64xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_12{{.*}} : (memref<1x1x64x256xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_13{{.*}} : (memref<1x1x64x256xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_14{{.*}} : (memref<1x1x64x256xbf16, 1 : i32>)
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c31{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.put{{.*}}@channel_15{{.*}} : (memref<1x1x64x256xbf16, 1 : i32>)
// CHECK: air.herd @herd_0
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_8{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_9{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_10{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: else
// CHECK-NEXT: air.channel.get{{.*}}@channel_11{{.*}} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_12{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_13{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_14{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
// CHECK: else
// CHECK-NEXT: air.channel.get{{.*}}@channel_15{{.*}} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
module {
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_4 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_5 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_6 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_8 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_9 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_10 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_11 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_12 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_13 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_14 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_15 [1, 1] {broadcast_shape = [4, 1]}
  func.func @func10(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048x2048xbf16>, %arg2: memref<2048x2048xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c8, %arg6=%c8) attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c192 = arith.constant 192 : index
        %c128 = arith.constant 128 : index
        %c16384 = arith.constant 16384 : index
        %c8_0 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c256 = arith.constant 256 : index
        %c2048 = arith.constant 2048 : index
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c31 = arith.constant 31 : index
        %async_token, %results = air.execute -> (memref<1x1x16x8x8x4xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
        }
        %async_token_1, %results_2 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
        }
        %async_token_3, %results_4 = air.execute -> (memref<1x1x64x256xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x256xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x256xbf16, 1 : i32>
        }
        %async_token_5, %results_6 = air.execute -> (memref<1x1x256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x256x64xbf16, 1 : i32>
        }
        %2 = air.channel.put async [%async_token_5]  @channel_0[] (%results_6[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 10 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        %3 = air.channel.put async [%async_token_5]  @channel_1[] (%results_6[%c0, %c0, %c0, %c0, %c64, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 11 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        %4 = air.channel.put async [%async_token_5]  @channel_2[] (%results_6[%c0, %c0, %c0, %c0, %c128, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 12 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        %5 = air.channel.put async [%async_token_5]  @channel_3[] (%results_6[%c0, %c0, %c0, %c0, %c192, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 13 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        %6 = air.channel.put async [%async_token_3]  @channel_4[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 14 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        %7 = air.channel.put async [%async_token_3]  @channel_5[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c64] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 15 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        %8 = air.channel.put async [%async_token_3]  @channel_6[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c128] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 16 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        %9 = air.channel.put async [%async_token_3]  @channel_7[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c192] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 17 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        %10 = air.herd @herd_0 async [%async_token, %async_token_1, %async_token_3, %async_token_5]  tile (%arg7, %arg8) in (%arg9=%c4, %arg10=%c4) args(%arg11=%results_2, %arg12=%results) : memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "mm.o"} {
          %12 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
            %14 = air.channel.get async  @channel_0[%arg7, %arg8] (%arg11[] [] []) {id = 18 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
            affine.yield %14 : !air.async.token
          } else {
            %14 = affine.if #set1()[%arg7, %arg8] -> !air.async.token {
              %15 = air.channel.get async  @channel_1[%arg7, %arg8] (%arg11[] [] []) {id = 19 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
              affine.yield %15 : !air.async.token
            } else {
              %15 = affine.if #set2()[%arg7, %arg8] -> !air.async.token {
                %16 = air.channel.get async  @channel_2[%arg7, %arg8] (%arg11[] [] []) {id = 20 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %16 : !air.async.token
              } else {
                %16 = air.channel.get async  @channel_3[%arg7, %arg8] (%arg11[] [] []) {id = 21 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %16 : !air.async.token
              }
              affine.yield %15 : !air.async.token
            }
            affine.yield %14 : !air.async.token
          }
          %13 = affine.if #set3()[%arg7, %arg8] -> !air.async.token {
            %14 = air.channel.get async  @channel_4[%arg7, %arg8] (%arg12[] [] []) {id = 22 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
            affine.yield %14 : !air.async.token
          } else {
            %14 = affine.if #set4()[%arg7, %arg8] -> !air.async.token {
              %15 = air.channel.get async  @channel_5[%arg7, %arg8] (%arg12[] [] []) {id = 23 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
              affine.yield %15 : !air.async.token
            } else {
              %15 = affine.if #set5()[%arg7, %arg8] -> !air.async.token {
                %16 = air.channel.get async  @channel_6[%arg7, %arg8] (%arg12[] [] []) {id = 24 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
                affine.yield %16 : !air.async.token
              } else {
                %16 = air.channel.get async  @channel_7[%arg7, %arg8] (%arg12[] [] []) {id = 25 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
                affine.yield %16 : !air.async.token
              }
              affine.yield %15 : !air.async.token
            }
            affine.yield %14 : !air.async.token
          }
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_5]  @channel_8[] (%results_6[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 28 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_5]  @channel_9[] (%results_6[%c0, %c0, %c0, %c0, %c64, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 29 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_5]  @channel_10[] (%results_6[%c0, %c0, %c0, %c0, %c128, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 30 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_5]  @channel_11[] (%results_6[%c0, %c0, %c0, %c0, %c192, %c0] [%c1, %c1, %c8_0, %c16, %c4, %c8_0] [%c16384, %c16384, %c8_0, %c256, %c64, %c1]) {id = 31 : i32} : (memref<1x1x256x64xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_3]  @channel_12[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 32 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_3]  @channel_13[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c64] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 33 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_3]  @channel_14[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c128] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 34 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        }
        scf.for %arg7 = %c1 to %c31 step %c1 {
          %12 = air.channel.put async [%async_token_3]  @channel_15[] (%results_4[%c0, %c0, %c0, %c0, %c0, %c192] [%c1, %c1, %c16, %c8_0, %c8_0, %c4] [%c16384, %c16384, %c4, %c2048, %c256, %c1]) {id = 35 : i32} : (memref<1x1x64x256xbf16, 1 : i32>)
        }
        %11 = air.herd @herd_0 async [%10]  tile (%arg7, %arg8) in (%arg9=%c4, %arg10=%c4) args(%arg11=%results_2, %arg12=%results) : memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32> attributes {id = 4 : i32, link_with = "mm.o"} {
          %c1_11 = arith.constant 1 : index
          %c31_12 = arith.constant 31 : index
          scf.for %arg13 = %c1_11 to %c31_12 step %c1_11 {
            %12 = affine.if #set()[%arg7, %arg8] -> !air.async.token {
              %14 = air.channel.get async  @channel_8[%arg7, %arg8] (%arg11[] [] []) {id = 36 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
              affine.yield %14 : !air.async.token
            } else {
              %14 = affine.if #set1()[%arg7, %arg8] -> !air.async.token {
                %15 = air.channel.get async  @channel_9[%arg7, %arg8] (%arg11[] [] []) {id = 37 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                affine.yield %15 : !air.async.token
              } else {
                %15 = affine.if #set2()[%arg7, %arg8] -> !air.async.token {
                  %16 = air.channel.get async  @channel_10[%arg7, %arg8] (%arg11[] [] []) {id = 38 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                  affine.yield %16 : !air.async.token
                } else {
                  %16 = air.channel.get async  @channel_11[%arg7, %arg8] (%arg11[] [] []) {id = 39 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
                  affine.yield %16 : !air.async.token
                }
                affine.yield %15 : !air.async.token
              }
              affine.yield %14 : !air.async.token
            }
            %13 = affine.if #set3()[%arg7, %arg8] -> !air.async.token {
              %14 = air.channel.get async  @channel_12[%arg7, %arg8] (%arg12[] [] []) {id = 40 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
              affine.yield %14 : !air.async.token
            } else {
              %14 = affine.if #set4()[%arg7, %arg8] -> !air.async.token {
                %15 = air.channel.get async  @channel_13[%arg7, %arg8] (%arg12[] [] []) {id = 41 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
                affine.yield %15 : !air.async.token
              } else {
                %15 = affine.if #set5()[%arg7, %arg8] -> !air.async.token {
                  %16 = air.channel.get async  @channel_14[%arg7, %arg8] (%arg12[] [] []) {id = 42 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
                  affine.yield %16 : !air.async.token
                } else {
                  %16 = air.channel.get async  @channel_15[%arg7, %arg8] (%arg12[] [] []) {id = 43 : i32} : (memref<1x1x16x8x8x4xbf16, 2 : i32>)
                  affine.yield %16 : !air.async.token
                }
                affine.yield %15 : !air.async.token
              }
              affine.yield %14 : !air.async.token
            }
          }
        }
        %async_token_7 = air.execute [%11] {
          memref.dealloc %results_6 : memref<1x1x256x64xbf16, 1 : i32>
        }
        %async_token_8 = air.execute [%11] {
          memref.dealloc %results_4 : memref<1x1x64x256xbf16, 1 : i32>
        }
        %async_token_9 = air.execute [%11] {
          memref.dealloc %results_2 : memref<1x1x8x16x4x8xbf16, 2 : i32>
        }
        %async_token_10 = air.execute [%11] {
          memref.dealloc %results : memref<1x1x16x8x8x4xbf16, 2 : i32>
        }
      }
    }
    return
  }
}

// -----

// Fusing scf.for and scf.parallel loop nest containing multiple air.channel.put/get ops.

// AGGL1-LABEL: @func11
// AGGL1: air.launch
// AGGL1: scf.for
// AGGL1: air.channel.put{{.*}}@channel_4
// AGGL1: scf.yield
// AGGL1: scf.for
// AGGL1: air.channel.put{{.*}}@channel_5
// AGGL1: scf.yield
// AGGL1: air.segment
// AGGL1: air.herd
// AGGL1-NEXT: air.channel.get{{.*}}@channel_6
// AGGL1-NEXT: air.channel.get{{.*}}@channel_6
// AGGL1: scf.for
// AGGL1-NEXT: %[[TOK0:.*]] = air.channel.get{{.*}}@channel_4
// AGGL1-NEXT: %[[TOK1:.*]] = air.channel.get{{.*}}@channel_5
// AGGL1-NEXT: air.wait_all async [%[[TOK0]], %[[TOK1]]] 
// AGGL1-NEXT: scf.parallel
// AGGL1: air.channel.put{{.*}}@channel_6
// AGGL1: air.channel.put{{.*}}@channel_6
// AGGL1: scf.reduce
// AGGL1: scf.yield
// AGGL1: air.herd
// AGGL1: scf.for
// AGGL1-NEXT: air.channel.get{{.*}}@channel_6
// AGGL1-NEXT: air.channel.get{{.*}}@channel_6
// AGGL1: air.herd
// AGGL1-NEXT: air.channel.get{{.*}}@channel_6
// AGGL1-NEXT: air.channel.get{{.*}}@channel_6

// AGGRESSIVE-LABEL: @func11
// AGGRESSIVE: air.launch
// AGGRESSIVE: scf.for
// AGGRESSIVE: air.channel.put{{.*}}@channel_4
// AGGRESSIVE: air.channel.put{{.*}}@channel_4
// AGGRESSIVE: scf.yield
// AGGRESSIVE: air.segment
// AGGRESSIVE: air.herd
// AGGRESSIVE-NEXT: air.channel.get{{.*}}@channel_6
// AGGRESSIVE-NEXT: air.channel.get{{.*}}@channel_6
// AGGRESSIVE: scf.for
// AGGRESSIVE-NEXT: %[[TOK0:.*]] = air.channel.get{{.*}}@channel_4
// AGGRESSIVE-NEXT: %[[TOK1:.*]] = air.channel.get{{.*}}@channel_4
// AGGRESSIVE-NEXT: air.wait_all async [%[[TOK0]], %[[TOK1]]] 
// AGGRESSIVE-NEXT: scf.parallel
// AGGRESSIVE: air.channel.put{{.*}}@channel_6
// AGGRESSIVE: air.channel.put{{.*}}@channel_6
// AGGRESSIVE: scf.reduce
// AGGRESSIVE: scf.yield
// AGGRESSIVE: air.herd
// AGGRESSIVE: scf.for
// AGGRESSIVE-NEXT: air.channel.get{{.*}}@channel_6
// AGGRESSIVE-NEXT: air.channel.get{{.*}}@channel_6
// AGGRESSIVE: air.herd
// AGGRESSIVE-NEXT: air.channel.get{{.*}}@channel_6
// AGGRESSIVE-NEXT: air.channel.get{{.*}}@channel_6

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [2, 2]
  air.channel @channel_3 [2, 2]
  air.channel @channel_4 [1, 1]
  air.channel @channel_5 [1, 1]
  air.channel @channel_6 [2, 2]
  air.channel @channel_7 [2, 2]
  air.channel @channel_8 [1, 1]
  air.channel @channel_9 [1, 1]
  air.channel @channel_12 [2, 2]
  air.channel @channel_13 [2, 2]
  func.func @func11(%arg0: memref<1024x512xi32>, %arg1: memref<512x1024xi32>) {
    %c16 = arith.constant 16 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c16, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1) : memref<1024x512xi32>, memref<512x1024xi32> attributes {id = 1 : i32} {
      %c480 = arith.constant 480 : index
      %c15 = arith.constant 15 : index
      %c1024 = arith.constant 1024 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %8 = affine.apply #map()[%arg2]
        air.execute_terminator %8 : index
      }
      %1 = air.channel.put async [%async_token]  @channel_0[] (%arg6[%results, %c0] [%c64, %c32] [%c512, %c1]) {id = 1 : i32} : (memref<1024x512xi32>)
      %async_token_0, %results_1 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %2 = air.channel.put async [%async_token_0]  @channel_1[] (%arg7[%c0, %results_1] [%c32, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<512x1024xi32>)
      %async_token_2, %results_3 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg2]
        air.execute_terminator %8 : index
      }
      %3 = scf.for %arg8 = %c1 to %c15 step %c1 iter_args(%arg9 = %async_token_2) -> (!air.async.token) {
        %async_token_10, %results_11 = air.execute [%arg9] -> (index) {
          %9 = affine.apply #map1()[%arg8]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.put async [%async_token_10]  @channel_4[] (%arg6[%results_3, %results_11] [%c64, %c32] [%c512, %c1]) {id = 3 : i32} : (memref<1024x512xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_4, %results_5 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %4 = scf.for %arg8 = %c1 to %c15 step %c1 iter_args(%arg9 = %async_token_4) -> (!air.async.token) {
        %async_token_10, %results_11 = air.execute [%arg9] -> (index) {
          %9 = affine.apply #map1()[%arg8]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.put async [%async_token_10]  @channel_5[] (%arg7[%results_11, %results_5] [%c32, %c64] [%c1024, %c1]) {id = 4 : i32} : (memref<512x1024xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_6, %results_7 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg2]
        air.execute_terminator %8 : index
      }
      %5 = air.channel.put async [%async_token_6]  @channel_8[] (%arg6[%results_7, %c480] [%c64, %c32] [%c512, %c1]) {id = 5 : i32} : (memref<1024x512xi32>)
      %async_token_8, %results_9 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %6 = air.channel.put async [%async_token_8]  @channel_9[] (%arg7[%c480, %results_9] [%c32, %c64] [%c1024, %c1]) {id = 6 : i32} : (memref<512x1024xi32>)
      %7 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c2048 = arith.constant 2048 : index
        %c8 = arith.constant 8 : index
        %c128 = arith.constant 128 : index
        %c4 = arith.constant 4 : index
        %c32_10 = arith.constant 32 : index
        %c64_11 = arith.constant 64 : index
        %c512_12 = arith.constant 512 : index
        %c0_13 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_14 = arith.constant 1 : index
        %c15_15 = arith.constant 15 : index
        %async_token_16, %results_17 = air.execute -> (memref<1x1x8x4x8x4xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<1x1x4x8x4x8xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<1x1x32x64xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x32x64xi32, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x32x64xi32, 1 : i32>
        }
        %async_token_22, %results_23 = air.execute -> (memref<1x1x64x32xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x1x64x32xi32, 1 : i32>
          air.execute_terminator %alloc : memref<1x1x64x32xi32, 1 : i32>
        }
        %8 = air.channel.get async [%async_token_22]  @channel_0[] (%results_23[] [] []) {id = 10 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        %9 = air.channel.get async [%async_token_20]  @channel_1[] (%results_21[] [] []) {id = 11 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        %10 = scf.parallel (%arg8, %arg9) = (%c0_13, %c0_13) to (%c2, %c2) step (%c1_14, %c1_14) init (%8) -> !air.async.token {
          %async_token_28, %results_29 = air.execute -> (index) {
            %22 = affine.apply #map1()[%arg8]
            air.execute_terminator %22 : index
          }
          %21 = air.channel.put async [%8, %async_token_28]  @channel_2[%arg8, %arg9] (%results_23[%c0_13, %c0_13, %c0_13, %c0_13, %results_29, %c0_13] [%c1_14, %c1_14, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32_10, %c1_14]) {id = 12 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
          scf.reduce(%21 : !air.async.token) {
          ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
            %22 = air.wait_all async [%arg10, %arg11] 
            scf.reduce.return %22 : !air.async.token
          }
        }
        %11 = scf.parallel (%arg8, %arg9) = (%c0_13, %c0_13) to (%c2, %c2) step (%c1_14, %c1_14) init (%9) -> !air.async.token {
          %async_token_28, %results_29 = air.execute -> (index) {
            %22 = affine.apply #map1()[%arg9]
            air.execute_terminator %22 : index
          }
          %21 = air.channel.put async [%9, %async_token_28]  @channel_3[%arg8, %arg9] (%results_21[%c0_13, %c0_13, %c0_13, %c0_13, %c0_13, %results_29] [%c1_14, %c1_14, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512_12, %c64_11, %c1_14]) {id = 13 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
          scf.reduce(%21 : !air.async.token) {
          ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
            %22 = air.wait_all async [%arg10, %arg11] 
            scf.reduce.return %22 : !air.async.token
          }
        }
        %12 = air.herd @herd_0 async [%async_token_16, %async_token_18]  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) args(%arg12=%results_19, %arg13=%results_17) : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32> attributes {id = 3 : i32} {
          %21 = air.channel.get async  @channel_2[%arg8, %arg9] (%arg12[] [] []) {id = 14 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
          %22 = air.channel.get async  @channel_3[%arg8, %arg9] (%arg13[] [] []) {id = 15 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
        }
        %13 = scf.for %arg8 = %c1_14 to %c15_15 step %c1_14 iter_args(%arg9 = %8) -> (!air.async.token) {
          %21 = air.channel.get async [%arg9]  @channel_4[] (%results_23[] [] []) {id = 16 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
          %22 = scf.parallel (%arg10, %arg11) = (%c0_13, %c0_13) to (%c2, %c2) step (%c1_14, %c1_14) init (%21) -> !air.async.token {
            %async_token_28, %results_29 = air.execute -> (index) {
              %24 = affine.apply #map1()[%arg10]
              air.execute_terminator %24 : index
            }
            %23 = air.channel.put async [%21, %async_token_28]  @channel_6[%arg10, %arg11] (%results_23[%c0_13, %c0_13, %c0_13, %c0_13, %results_29, %c0_13] [%c1_14, %c1_14, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32_10, %c1_14]) {id = 18 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
            scf.reduce(%23 : !air.async.token) {
            ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
              %24 = air.wait_all async [%arg12, %arg13] 
              scf.reduce.return %24 : !air.async.token
            }
          }
          scf.yield %22 : !air.async.token
        }
        %14 = scf.for %arg8 = %c1_14 to %c15_15 step %c1_14 iter_args(%arg9 = %9) -> (!air.async.token) {
          %21 = air.channel.get async [%arg9]  @channel_5[] (%results_21[] [] []) {id = 17 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
          %22 = scf.parallel (%arg10, %arg11) = (%c0_13, %c0_13) to (%c2, %c2) step (%c1_14, %c1_14) init (%21) -> !air.async.token {
            %async_token_28, %results_29 = air.execute -> (index) {
              %24 = affine.apply #map1()[%arg11]
              air.execute_terminator %24 : index
            }
            %23 = air.channel.put async [%21, %async_token_28]  @channel_7[%arg10, %arg11] (%results_21[%c0_13, %c0_13, %c0_13, %c0_13, %c0_13, %results_29] [%c1_14, %c1_14, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512_12, %c64_11, %c1_14]) {id = 19 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
            scf.reduce(%23 : !air.async.token) {
            ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
              %24 = air.wait_all async [%arg12, %arg13] 
              scf.reduce.return %24 : !air.async.token
            }
          }
          scf.yield %22 : !air.async.token
        }
        %15 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) args(%arg12=%results_19, %arg13=%results_17) : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32> attributes {id = 4 : i32} {
          %c15_28 = arith.constant 15 : index
          %c1_29 = arith.constant 1 : index
          scf.for %arg14 = %c1_29 to %c15_28 step %c1_29 {
            %21 = air.channel.get async  @channel_6[%arg8, %arg9] (%arg12[] [] []) {id = 20 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
            %22 = air.channel.get async  @channel_7[%arg8, %arg9] (%arg13[] [] []) {id = 21 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
          }
        }
        %16 = air.channel.get async [%10, %13]  @channel_8[] (%results_23[] [] []) {id = 22 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
        %17 = air.channel.get async [%async_token_20, %14]  @channel_9[] (%results_21[] [] []) {id = 23 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
        %18 = scf.parallel (%arg8, %arg9) = (%c0_13, %c0_13) to (%c2, %c2) step (%c1_14, %c1_14) init (%16) -> !air.async.token {
          %async_token_28, %results_29 = air.execute -> (index) {
            %22 = affine.apply #map1()[%arg8]
            air.execute_terminator %22 : index
          }
          %21 = air.channel.put async [%16, %async_token_28]  @channel_12[%arg8, %arg9] (%results_23[%c0_13, %c0_13, %c0_13, %c0_13, %results_29, %c0_13] [%c1_14, %c1_14, %c4, %c8, %c4, %c8] [%c2048, %c2048, %c8, %c128, %c32_10, %c1_14]) {id = 26 : i32} : (memref<1x1x64x32xi32, 1 : i32>)
          scf.reduce(%21 : !air.async.token) {
          ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
            %22 = air.wait_all async [%arg10, %arg11] 
            scf.reduce.return %22 : !air.async.token
          }
        }
        %19 = scf.parallel (%arg8, %arg9) = (%c0_13, %c0_13) to (%c2, %c2) step (%c1_14, %c1_14) init (%17) -> !air.async.token {
          %async_token_28, %results_29 = air.execute -> (index) {
            %22 = affine.apply #map1()[%arg9]
            air.execute_terminator %22 : index
          }
          %21 = air.channel.put async [%17, %async_token_28]  @channel_13[%arg8, %arg9] (%results_21[%c0_13, %c0_13, %c0_13, %c0_13, %c0_13, %results_29] [%c1_14, %c1_14, %c8, %c4, %c8, %c4] [%c2048, %c2048, %c4, %c512_12, %c64_11, %c1_14]) {id = 27 : i32} : (memref<1x1x32x64xi32, 1 : i32>)
          scf.reduce(%21 : !air.async.token) {
          ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
            %22 = air.wait_all async [%arg10, %arg11] 
            scf.reduce.return %22 : !air.async.token
          }
        }
        %20 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) args(%arg12=%results_19, %arg13=%results_17) : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32> attributes {id = 5 : i32} {
          %21 = air.channel.get async  @channel_12[%arg8, %arg9] (%arg12[] [] []) {id = 30 : i32} : (memref<1x1x4x8x4x8xi32, 2 : i32>)
          %22 = air.channel.get async  @channel_13[%arg8, %arg9] (%arg13[] [] []) {id = 31 : i32} : (memref<1x1x8x4x8x4xi32, 2 : i32>)
        }
        %async_token_24 = air.execute [%18] {
          memref.dealloc %results_23 : memref<1x1x64x32xi32, 1 : i32>
        }
        %async_token_25 = air.execute {
          memref.dealloc %results_21 : memref<1x1x32x64xi32, 1 : i32>
        }
        %async_token_26 = air.execute {
          memref.dealloc %results_19 : memref<1x1x4x8x4x8xi32, 2 : i32>
        }
        %async_token_27 = air.execute {
          memref.dealloc %results_17 : memref<1x1x8x4x8x4xi32, 2 : i32>
        }
      }
    }
    return
  }
}

// -----

// Fusing at the inner-most loop in the control loop nest.

// CHECK-LABEL: @func12
// CHECK: %[[FOR0:.*]] = scf.for
// CHECK:   %[[FOR1:.*]] = scf.for
// CHECK:     %[[FOR2:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c16{{.*}} step %c1{{.*}}
// CHECK:       %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_8
// CHECK:       scf.yield %[[PUT0]] : !air.async.token
// CHECK:     }
// CHECK:     scf.yield %[[FOR2]] : !air.async.token
// CHECK:   }
// CHECK:   scf.yield %[[FOR1]] : !air.async.token
// CHECK: }
// CHECK: %[[FOR0:.*]] = scf.for
// CHECK:   %[[FOR1:.*]] = scf.for
// CHECK:     %[[FOR2:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c16{{.*}} step %c1{{.*}}
// CHECK:       %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_9
// CHECK:       scf.yield %[[PUT0]] : !air.async.token
// CHECK:     }
// CHECK:     scf.yield %[[FOR2]] : !air.async.token
// CHECK:   }
// CHECK:   scf.yield %[[FOR1]] : !air.async.token
// CHECK: }
// CHECK: air.herd @herd_0
// CHECK: %[[FOR0:.*]] = scf.for
// CHECK:   %[[FOR1:.*]] = scf.for
// CHECK: affine.if
// CHECK-NEXT: air.channel.get async{{.*}}@channel_8
// CHECK-NEXT: affine.yield
// CHECK-NEXT: else
// CHECK-NEXT: air.channel.get async{{.*}}@channel_9
// CHECK-NEXT: affine.yield
// CHECK-NEXT: }
// CHECK: scf.for %{{.*}} = %c1{{.*}} to %c15{{.*}} step %c1{{.*}}
// CHECK: affine.if
// CHECK-NEXT: air.channel.get async{{.*}}@channel_8
// CHECK-NEXT: affine.yield
// CHECK-NEXT: else
// CHECK-NEXT: air.channel.get async{{.*}}@channel_9
// CHECK-NEXT: affine.yield
// CHECK-NEXT: }
// CHECK: }
// CHECK: affine.if
// CHECK-NEXT: air.channel.get async{{.*}}@channel_8
// CHECK-NEXT: affine.yield
// CHECK-NEXT: else
// CHECK-NEXT: air.channel.get async{{.*}}@channel_9
// CHECK-NEXT: affine.yield
// CHECK-NEXT: }

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_8 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_9 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_16 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_17 [1, 1] {broadcast_shape = [1, 4]}
  func.func @func12(%arg0: memref<1x1x4x8x4x8xbf16, 2 : i32>, %arg1: memref<8x16x32x32xbf16, 1 : i32>, %arg2: memref<8x16x32x32xbf16, 1 : i32>) {
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c15 = arith.constant 15 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16384 = arith.constant 16384 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = air.wait_all async 
    %1 = scf.for %arg3 = %c0 to %c8 step %c4 iter_args(%arg4 = %0) -> (!air.async.token) {
      %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_0[] (%arg2[%arg3, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 6 : i32} : (memref<8x16x32x32xbf16, 1 : i32>)
        scf.yield %9 : !air.async.token
      }
      scf.yield %8 : !air.async.token
    }
    %2 = scf.for %arg3 = %c0 to %c8 step %c4 iter_args(%arg4 = %0) -> (!air.async.token) {
      %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_1[] (%arg2[%arg3, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 7 : i32} : (memref<8x16x32x32xbf16, 1 : i32>)
        scf.yield %9 : !air.async.token
      }
      scf.yield %8 : !air.async.token
    }
    %3 = scf.for %arg3 = %c0 to %c8 step %c4 iter_args(%arg4 = %0) -> (!air.async.token) {
      %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (!air.async.token) {
        %9 = scf.for %arg7 = %c1 to %c15 step %c1 iter_args(%arg8 = %arg6) -> (!air.async.token) {
          %10 = air.channel.put async [%arg8]  @channel_8[] (%arg2[%arg3, %arg7, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 22 : i32} : (memref<8x16x32x32xbf16, 1 : i32>)
          scf.yield %10 : !air.async.token
        }
        scf.yield %9 : !air.async.token
      }
      scf.yield %8 : !air.async.token
    }
    %4 = scf.for %arg3 = %c0 to %c8 step %c4 iter_args(%arg4 = %0) -> (!air.async.token) {
      %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (!air.async.token) {
        %9 = scf.for %arg7 = %c1 to %c15 step %c1 iter_args(%arg8 = %arg6) -> (!air.async.token) {
          %10 = air.channel.put async [%arg8]  @channel_9[] (%arg2[%arg3, %arg7, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 23 : i32} : (memref<8x16x32x32xbf16, 1 : i32>)
          scf.yield %10 : !air.async.token
        }
        scf.yield %9 : !air.async.token
      }
      scf.yield %8 : !air.async.token
    }
    %5 = air.herd @herd_0 async  tile (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) args(%arg7=%arg0) : memref<1x1x4x8x4x8xbf16, 2 : i32> attributes {id = 5 : i32, link_with = "mm.o"} {
      %c15_0 = arith.constant 15 : index
      %c1_1 = arith.constant 1 : index
      %c4_2 = arith.constant 4 : index
      %c0_3 = arith.constant 0 : index
      %c8_4 = arith.constant 8 : index
      %8 = air.wait_all async 
      %9 = scf.for %arg8 = %c0_3 to %c8_4 step %c4_2 iter_args(%arg9 = %8) -> (!air.async.token) {
        %10 = scf.for %arg10 = %c0_3 to %c8_4 step %c4_2 iter_args(%arg11 = %arg9) -> (!air.async.token) {
          %11 = affine.if #set()[%arg3, %arg4] -> !air.async.token {
            %14 = air.channel.get async  @channel_0[%arg3, %arg4] (%arg7[] [] []) {id = 14 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>)
            affine.yield %14 : !air.async.token
          } else {
            %14 = air.channel.get async  @channel_1[%arg3, %arg4] (%arg7[] [] []) {id = 15 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>)
            affine.yield %14 : !air.async.token
          }
          scf.for %arg12 = %c1_1 to %c15_0 step %c1_1 {
            %14 = affine.if #set()[%arg3, %arg4] -> !air.async.token {
              %15 = air.channel.get async  @channel_8[%arg3, %arg4] (%arg7[] [] []) {id = 30 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>)
              affine.yield %15 : !air.async.token
            } else {
              %15 = air.channel.get async  @channel_9[%arg3, %arg4] (%arg7[] [] []) {id = 31 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>)
              affine.yield %15 : !air.async.token
            }
          }
          %12 = affine.if #set()[%arg3, %arg4] -> !air.async.token {
            %14 = air.channel.get async  @channel_16[%arg3, %arg4] (%arg7[] [] []) {id = 47 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>)
            affine.yield %14 : !air.async.token
          } else {
            %14 = air.channel.get async  @channel_17[%arg3, %arg4] (%arg7[] [] []) {id = 48 : i32} : (memref<1x1x4x8x4x8xbf16, 2 : i32>)
            affine.yield %14 : !air.async.token
          }
          %13 = air.wait_all async 
          scf.yield %13 : !air.async.token
        }
        scf.yield %10 : !air.async.token
      }
    }
    %6 = scf.for %arg3 = %c0 to %c8 step %c4 iter_args(%arg4 = %0) -> (!air.async.token) {
      %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_16[] (%arg2[%arg3, %c15, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 38 : i32} : (memref<8x16x32x32xbf16, 1 : i32>)
        scf.yield %9 : !air.async.token
      }
      scf.yield %8 : !air.async.token
    }
    %7 = scf.for %arg3 = %c0 to %c8 step %c4 iter_args(%arg4 = %0) -> (!air.async.token) {
      %8 = scf.for %arg5 = %c0 to %c8 step %c4 iter_args(%arg6 = %arg4) -> (!air.async.token) {
        %9 = air.channel.put async [%arg6]  @channel_17[] (%arg2[%arg3, %c15, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c16384, %c1024, %c8, %c128, %c32, %c1]) {id = 39 : i32} : (memref<8x16x32x32xbf16, 1 : i32>)
        scf.yield %9 : !air.async.token
      }
      scf.yield %8 : !air.async.token
    }
    return
  }
}

// -----

// No-for-loop (NFL) mode: preserve async dependencies of multiple fused channe put/get ops, by fusing their parent regions.

// CHECK-LABEL: func13
// CHECK: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put async {{.*}} @channel_0
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK: %[[FOR2:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put async {{.*}} @channel_1
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK: air.segment
// CHECK: %[[FOR3:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.get async {{.*}} @channel_0
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK: %[[FOR4:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// CHECK-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// CHECK-NEXT: air.channel.get async {{.*}} @channel_1
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }

// AGGRESSIVE-LABEL: func13
// AGGRESSIVE: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: affine.apply
// AGGRESSIVE-NEXT: air.channel.put async {{.*}} @channel_0
// AGGRESSIVE-NEXT: affine.apply
// AGGRESSIVE-NEXT: affine.apply
// AGGRESSIVE-NEXT: air.channel.put async {{.*}} @channel_0
// AGGRESSIVE-NEXT: air.wait_all
// AGGRESSIVE-NEXT: scf.yield
// AGGRESSIVE-NEXT: }
// AGGRESSIVE-NEXT: scf.yield
// AGGRESSIVE-NEXT: }
// AGGRESSIVE: air.segment
// AGGRESSIVE: %[[FOR2:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// AGGRESSIVE-NEXT: air.channel.get async {{.*}} @channel_0
// AGGRESSIVE-NEXT: air.channel.get async {{.*}} @channel_0
// AGGRESSIVE-NEXT: air.wait_all
// AGGRESSIVE-NEXT: scf.yield
// AGGRESSIVE-NEXT: }
// AGGRESSIVE-NEXT: scf.yield
// AGGRESSIVE-NEXT: }

// AGGL1-LABEL: func13
// AGGL1: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// AGGL1-NEXT: affine.apply
// AGGL1-NEXT: air.channel.put async {{.*}} @channel_0
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1: %[[FOR2:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// AGGL1-NEXT: affine.apply
// AGGL1-NEXT: air.channel.put async {{.*}} @channel_1
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1: air.segment
// AGGL1: %[[FOR3:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// AGGL1-NEXT: air.channel.get async {{.*}} @channel_0
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1: %[[FOR4:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1{{.*}}
// AGGL1-NEXT: %[[FOR1:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1{{.*}}
// AGGL1-NEXT: air.channel.get async {{.*}} @channel_1
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }
// AGGL1-NEXT: scf.yield
// AGGL1-NEXT: }

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d0, d4, d3, d6, d7)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set7 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 3 == 0)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_12 [1, 1]
  air.channel @channel_13 [1, 1]
  func.func @func13(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>, %arg3: memref<512x512xbf16>, %arg4: memref<512x512xbf16>, %arg5: memref<512x512xbf16>, %arg6: memref<512x512xbf16>, %arg7: memref<512x512xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) args(%arg12=%arg0, %arg13=%arg1, %arg14=%arg2, %arg15=%arg3) : memref<512x512xbf16>, memref<512x512xbf16>, memref<512x512xbf16>, memref<512x512xbf16> attributes {id = 1 : i32} {
      %c256 = arith.constant 256 : index
      %c65536 = arith.constant 65536 : index
      %c512 = arith.constant 512 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %1 = affine.apply #map()[%arg8]
      %2 = air.wait_all async 
      %3 = scf.for %arg16 = %c0 to %c4 step %c1 iter_args(%arg17 = %2) -> (!air.async.token) {
        %14 = affine.apply #map1()[%arg16]
        %15 = air.channel.put async [%arg17]  @channel_0[] (%arg12[%c0, %c0, %1, %14] [%c4, %c1, %c64, %c128] [%c32768, %c128, %c512, %c1]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield %15 : !air.async.token
      }
      %4 = affine.apply #map()[%arg9]
      %5 = air.wait_all async 
      %6 = scf.for %arg16 = %c0 to %c4 step %c1 iter_args(%arg17 = %5) -> (!air.async.token) {
        %14 = affine.apply #map1()[%arg16]
        %15 = air.channel.put async [%arg17]  @channel_1[] (%arg13[%c0, %c0, %14, %4] [%c1, %c4, %c128, %c64] [%c65536, %c64, %c512, %c1]) {id = 2 : i32} : (memref<512x512xbf16>)
        scf.yield %15 : !air.async.token
      }
      %7 = air.channel.get async  @channel_11[] (%arg15[%1, %4] [%c256, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
      %8 = air.wait_all async 
      %9 = scf.for %arg16 = %c0 to %c4 step %c1 iter_args(%arg17 = %8) -> (!air.async.token) {
        %14 = affine.apply #map1()[%arg16]
        %15 = air.channel.put async [%arg17]  @channel_12[] (%arg15[%c0, %c0, %1, %14] [%c4, %c1, %c64, %c128] [%c32768, %c128, %c512, %c1]) {id = 4 : i32} : (memref<512x512xbf16>)
        scf.yield %15 : !air.async.token
      }
      %10 = air.wait_all async 
      %11 = scf.for %arg16 = %c0 to %c4 step %c1 iter_args(%arg17 = %10) -> (!air.async.token) {
        %14 = affine.apply #map1()[%arg16]
        %15 = air.channel.put async [%arg17]  @channel_13[] (%arg14[%c0, %c0, %14, %4] [%c1, %c4, %c128, %c64] [%c65536, %c64, %c512, %c1]) {id = 5 : i32} : (memref<512x512xbf16>)
        scf.yield %15 : !air.async.token
      }
      %13 = air.segment @matmul_seg async  attributes {id = 2 : i32} {
        %c3 = arith.constant 3 : index
        %c2_0 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c8192 = arith.constant 8192 : index
        %c4096 = arith.constant 4096 : index
        %c16384 = arith.constant 16384 : index
        %c512_1 = arith.constant 512 : index
        %c32768_2 = arith.constant 32768 : index
        %c128_3 = arith.constant 128 : index
        %c64_4 = arith.constant 64 : index
        %c1_5 = arith.constant 1 : index
        %c0_6 = arith.constant 0 : index
        %c4_7 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<4x1x64x128xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<4x1x64x128xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<4x1x64x128xbf16, 1 : i32>
        }
        %async_token_8, %results_9 = air.execute -> (memref<1x4x128x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<1x4x128x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<1x4x128x64xbf16, 1 : i32>
        }
        %14 = air.wait_all async [%async_token, %async_token_8] 
        %15 = scf.for %arg16 = %c0_6 to %c4_7 step %c1_5 iter_args(%arg17 = %14) -> (!air.async.token) {
          %25 = air.channel.get async [%arg17]  @channel_0[] (%results[] [] []) {id = 7 : i32} : (memref<4x1x64x128xbf16, 1 : i32>)
          scf.yield %25 : !air.async.token
        }
        %16 = scf.for %arg16 = %c0_6 to %c4_7 step %c1_5 iter_args(%arg17 = %14) -> (!air.async.token) {
          %25 = air.channel.get async [%arg17]  @channel_1[] (%results_9[] [] []) {id = 8 : i32} : (memref<1x4x128x64xbf16, 1 : i32>)
          scf.yield %25 : !air.async.token
        }
        %19 = air.wait_all async [%async_token, %async_token_8] 
        %20 = scf.for %arg16 = %c0_6 to %c4_7 step %c1_5 iter_args(%arg17 = %19) -> (!air.async.token) {
          %25 = air.channel.get async [%arg17]  @channel_12[] (%results[] [] []) {id = 28 : i32} : (memref<4x1x64x128xbf16, 1 : i32>)
          scf.yield %25 : !air.async.token
        }
        %21 = scf.for %arg16 = %c0_6 to %c4_7 step %c1_5 iter_args(%arg17 = %19) -> (!air.async.token) {
          %25 = air.channel.get async [%arg17]  @channel_13[] (%results_9[] [] []) {id = 29 : i32} : (memref<1x4x128x64xbf16, 1 : i32>)
          scf.yield %25 : !air.async.token
        }
        %async_token_18 = air.execute {
          memref.dealloc %results : memref<4x1x64x128xbf16, 1 : i32>
        }
        %async_token_19 = air.execute {
          memref.dealloc %results_9 : memref<1x4x128x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}

