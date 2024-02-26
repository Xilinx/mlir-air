//===- fuse_channels.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels --split-input-file | FileCheck %s

// Have multiple channel put-get pairs share the same symbolic channels.
// CHECK-LABEL: func0
// CHECK: air.launch
// CHECK: air.channel.put @channel_0
// CHECK: air.channel.put @channel_0
// CHECK: air.segment
// CHECK: air.channel.get @channel_0
// CHECK: air.channel.get @channel_0
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

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
          air.herd_terminator
        }
        memref.dealloc %alloc_2 : memref<4x4xi32, 1>
        memref.dealloc %alloc_3 : memref<4x4xi32, 1>
        air.segment_terminator
      }
      memref.dealloc %alloc_0 : memref<4x4xi32>
      memref.dealloc %alloc_1 : memref<4x4xi32>
      air.launch_terminator
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
// CHECK: air.channel.put @channel_0
// CHECK: air.herd
// CHECK: air.channel.get @channel_2
// CHECK: scf.for
// CHECK: air.channel.get @channel_0
// CHECK: air.channel.get @channel_0
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

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

// CHECK-LABEL: func2
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.segment
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

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
        air.segment_terminator
      }
      %async_token_2 = air.execute [%1] {
        memref.dealloc %results : memref<4x4xi32>
      }
      %async_token_3 = air.execute [%2] {
        memref.dealloc %results_1 : memref<4x4xi32>
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL: func3
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

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
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
