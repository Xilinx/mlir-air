//===- broadcast_to_channel.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse --split-input-file | FileCheck %s

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
module {
// CHECK: air.channel @channel_0 [2, 1] {broadcast_shape = [2, 2]}
// CHECK-LABEL: @mmult
  func.func @mmult(%arg0: memref<512x512xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c8, %arg4=%c8) args(%arg5=%arg0) : memref<512x512xbf16> attributes {id = 3 : i32} {
// CHECK: %[[EVENT0:.*]] = air.segment async
      %1 = air.segment async  args(%arg6=%arg1, %arg7=%arg2, %arg8=%arg3, %arg9=%arg4, %arg10=%arg5) : index, index, index, index, memref<512x512xbf16> attributes {id = 2 : i32} {
// CHECK: %[[CONST1:.*]] = arith.constant 1 : index
// CHECK: %[[CONST2:.*]] = arith.constant 2 : index   
// CHECK: %[[CONST0:.*]] = arith.constant 0 : index   
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %async_token, %results = air.execute -> (index) {
          %4 = affine.apply #map()[%arg6]
          air.execute_terminator %4 : index
        } {id = 1 : i32}
        %2 = air.wait_all async [%async_token]  {id = 4 : i32}
// CHECK: %[[EVENT1:.*]] = scf.for
        %3 = scf.for %arg11 = %c0 to %c512 step %c64 iter_args(%arg12 = %2) -> (!air.async.token) {
          %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          } {id = 2 : i32}       
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE0:.*]]) = (%[[CONST0]]) to (%[[CONST2]]) step (%[[CONST1]])
// CHECK: %[[EVENT3:.*]] = scf.for
// CHECK: %[[EVENT4:.*]] = air.channel.put async{{.*}}@channel_0[%[[VALUE0]], %[[CONST0]]]
// CHECK: %[[EVENT5:.*]] = air.herd @herd_0 async{{.*}}tile (%[[VALUE2:.*]], %[[VALUE3:.*]]) in
// CHECK: %[[EVENT6:.*]] = air.channel.get async{{.*}}@channel_0[%[[VALUE2]], %[[VALUE3]]]
          %4 = air.herd @herd_0 async  tile (%arg13, %arg14) in (%arg15=%c2, %arg16=%c2) args(%arg17=%results_1) : memref<64x64xbf16, 1> attributes {id = 1 : i32} {
            %c1 = arith.constant 1 : index
            %c0_3 = arith.constant 0 : index
            %c64_4 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %async_token_5, %results_6 = air.execute -> (index) {
              %8 = affine.apply #map1()[%arg13]
              air.execute_terminator %8 : index
            } {id = 3 : i32}
            %6 = air.wait_all async [%async_token_5]  {id = 2 : i32}
            %7 = scf.for %arg18 = %c0_3 to %c64_4 step %c32 iter_args(%arg19 = %6) -> (!air.async.token) {
              %async_token_7, %results_8 = air.execute -> (memref<32x32xbf16, 2>) {
                %alloc = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %alloc : memref<32x32xbf16, 2>
              } {id = 4 : i32}
              %8 = air.dma_memcpy_nd async [%async_token_7, %arg19] (%results_8[] [] [], %arg17[%results_6, %arg18] [%c32, %c32] [%c64_4, %c1]) {broadcast_pattern = #set, id = 1 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              %async_token_9 = air.execute [%8] {
                memref.dealloc %results_8 : memref<32x32xbf16, 2>
              } {id = 5 : i32}
              %9 = air.wait_all async [%8]  {id = 1 : i32}
              scf.yield %9 : !air.async.token
            }
          }
          %async_token_2 = air.execute [%4] {
            memref.dealloc %results_1 : memref<64x64xbf16, 1>
          } {id = 6 : i32}
          %5 = air.wait_all async [%4]  {id = 3 : i32}
          scf.yield %5 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Broadcast to a 2D array of cores.

// CHECK-LABEL: func.func @conv
// CHECK: air.launch
// CHECK: air.segment
// CHECK-NOT: scf.parallel
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@channel_0[]
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: air.herd
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: affine.if
// CHECK-NEXT: air.channel.get{{.*}}@channel_0
// CHECK-NEXT: affine.yield
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: scf.yield

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  func.func @conv() {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %0 = air.launch async (%arg0, %arg1, %arg2) in (%arg3=%c3, %arg4=%c3, %arg5=%c16) attributes {id = 3 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %async_token, %results = air.execute -> (memref<3x3x32x4xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<3x3x32x4xi32, 1 : i32>
          air.execute_terminator %alloc : memref<3x3x32x4xi32, 1 : i32>
        } {id = 1 : i32}
        %2 = air.herd @herd_0 async [%async_token]  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c4) args(%arg10=%results) : memref<3x3x32x4xi32, 1 : i32> attributes {id = 1 : i32} {
          %c128 = arith.constant 128 : index
          %c384 = arith.constant 384 : index
          %c4_1 = arith.constant 4 : index
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c3_2 = arith.constant 3 : index
          %c1 = arith.constant 1 : index
          %3 = air.wait_all async  {id = 6 : i32}
          %4 = scf.for %arg11 = %c0 to %c3_2 step %c1 iter_args(%arg12 = %3) -> (!air.async.token) {
            %5 = scf.for %arg13 = %c0 to %c3_2 step %c1 iter_args(%arg14 = %arg12) -> (!air.async.token) {
              %7 = scf.for %arg15 = %c0 to %c32 step %c8 iter_args(%arg16 = %arg14) -> (!air.async.token) {
                %async_token_3, %results_4 = air.execute -> (memref<1x1x8x4xi32, 2 : i32>) {
                  %alloc = memref.alloc() : memref<1x1x8x4xi32, 2 : i32>
                  air.execute_terminator %alloc : memref<1x1x8x4xi32, 2 : i32>
                } {id = 2 : i32}
                %9 = affine.if #set()[%arg6, %arg7] -> !air.async.token {
                  %11 = air.dma_memcpy_nd async [%arg16, %async_token_3] (%results_4[] [] [], %arg10[%arg11, %arg13, %arg15, %c0] [%c1, %c1, %c8, %c4_1] [%c384, %c128, %c4_1, %c1]) {broadcast_set = #set, id = 1 : i32} : (memref<1x1x8x4xi32, 2 : i32>, memref<3x3x32x4xi32, 1 : i32>)
                  affine.yield %11 : !air.async.token
                } else {
                  %11 = air.wait_all async [%arg16, %async_token_3] 
                  affine.yield %11 : !air.async.token
                }
                %10 = air.wait_all async [%arg16, %9]  {id = 1 : i32}
                scf.yield %10 : !air.async.token
              }
              %8 = air.wait_all async [%arg14, %7]  {id = 3 : i32}
              scf.yield %8 : !air.async.token
            }
            %6 = air.wait_all async [%arg12, %5]  {id = 5 : i32}
            scf.yield %6 : !air.async.token
          }
        }
        %async_token_0 = air.execute [%2] {
          memref.dealloc %results : memref<3x3x32x4xi32, 1 : i32>
        } {id = 3 : i32}
      }
    }
    return
  }
}
