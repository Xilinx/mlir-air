//===- dma_to_channel_nested_for_in_herd.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse | FileCheck %s

// Hoisting external channel put/get op out of a herd with nested for loops

#map = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK-LABEL: func.func @sync
  func.func @sync(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %alloc, %alloc_0 : memref<64x64xi32> to memref<64x64xi32>
// CHECK: %[[EVENT0:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_0[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT1:.*]] = scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_1[%[[VALUE2]], %[[VALUE3]]]
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE4:.*]], %[[VALUE5:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_2[%[[VALUE4]], %[[VALUE5]]]
// CHECK: %[[EVENT3:.*]] = scf.parallel (%[[VALUE6:.*]], %[[VALUE7:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get @channel_3[%[[VALUE6]], %[[VALUE7]]]
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.for %newarg0 = %c0 to %c64 step %c32 {
        scf.for %newarg1 = %c0 to %c64 step %c32 {
            scf.for %arg9 = %c0 to %c64 step %c32 {
                %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
                %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
                %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
                air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%newarg0, %arg9] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                air.dma_memcpy_nd (%alloc_2[] [] [], %arg7[%arg9, %newarg1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                air.dma_memcpy_nd (%alloc_3[] [] [], %arg8[%newarg0, %newarg1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                linalg.matmul ins(%alloc_1, %alloc_2 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_3 : memref<32x32xi32, 2>)
                air.dma_memcpy_nd (%arg8[%newarg0, %newarg1] [%c32, %c32] [%c64, %c1], %alloc_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
                memref.dealloc %alloc_1 : memref<32x32xi32, 2>
                memref.dealloc %alloc_2 : memref<32x32xi32, 2>
                memref.dealloc %alloc_3 : memref<32x32xi32, 2>
            }
        }
      }
      air.herd_terminator
    }
    return %alloc_0 : memref<64x64xi32>
  }

// CHECK-LABEL: func.func @async
// CHECK: %[[EVENT0:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_4[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT1:.*]] = scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_5[%[[VALUE2]], %[[VALUE3]]]
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE4:.*]], %[[VALUE5:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_6[%[[VALUE4]], %[[VALUE5]]]
// CHECK: %[[EVENT3:.*]] = scf.parallel (%[[VALUE6:.*]], %[[VALUE7:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get async{{.*}}@channel_7[%[[VALUE6]], %[[VALUE7]]]  
  func.func @async(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<64x64xi32>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<64x64xi32> to memref<64x64xi32>
    }
    %0 = air.herd @herd_0 async [%async_token_3]  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%results_2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg9 = %c0 to %c64 step %c32 iter_args(%arg10 = %1) -> (!air.async.token) {
        %3 = scf.for %arg11 = %c0 to %c64 step %c32 iter_args(%arg12 = %arg10) -> (!air.async.token) {
          %4 = scf.for %arg13 = %c0 to %c64 step %c32 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_4, %results_5 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_6, %results_7 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_8, %results_9 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %5 = air.dma_memcpy_nd async [%async_token_4, %arg14] (%results_5[] [] [], %arg6[%arg9, %arg13] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %6 = air.dma_memcpy_nd async [%async_token_6, %arg14] (%results_7[] [] [], %arg7[%arg13, %arg11] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %7 = air.dma_memcpy_nd async [%async_token_8, %arg14] (%results_9[] [] [], %arg8[%arg9, %arg11] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %async_token_10 = air.execute [%7, %6, %5] {
              linalg.matmul ins(%results_5, %results_7 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_9 : memref<32x32xi32, 2>)
            }
            %8 = air.dma_memcpy_nd async [%async_token_10] (%arg8[%arg9, %arg11] [%c32, %c32] [%c64, %c1], %results_9[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
            %async_token_11 = air.execute [%async_token_10] {
              memref.dealloc %results_5 : memref<32x32xi32, 2>
            }
            %async_token_12 = air.execute [%async_token_10] {
              memref.dealloc %results_7 : memref<32x32xi32, 2>
            }
            %async_token_13 = air.execute [%8] {
              memref.dealloc %results_9 : memref<32x32xi32, 2>
            }
            scf.yield %8 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
      air.herd_terminator
    }
    return %results_2 : memref<64x64xi32>
  }

// CHECK-LABEL: func.func @legalize_memspace_sync
// CHECK: scf.parallel
// CHECK: scf.for{{.*}}%c0 to %c512 step %c64
// CHECK: scf.for{{.*}}%c0 to %c512 step %c64
// CHECK: scf.for{{.*}}%c0 to %c64 step %c32
// CHECK: scf.for{{.*}}%c0 to %c64 step %c32
// CHECK: scf.for{{.*}}%c0 to %c1024 step %c128
// CHECK: air.channel.put @channel_8
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK-NEXT: }
// CHECK: scf.parallel
// CHECK: scf.for{{.*}}%c0 to %c512 step %c64
// CHECK: scf.for{{.*}}%c0 to %c512 step %c64
// CHECK: air.channel.get @channel_9
// CHECK: }
// CHECK-NEXT: }
// CHECK: air.segment @segment_0
// CHECK: scf.parallel
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c64_1 step %c32_3
// CHECK: scf.for{{.*}}%c0_6 to %c64_1 step %c32_3
// CHECK: scf.for{{.*}}%c0_6 to %c1024_5 step %c128_4
// CHECK: air.channel.get @channel_8
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK-NEXT: }
// CHECK: scf.parallel
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: air.channel.put @channel_9
// CHECK: }
// CHECK-NEXT: }
// CHECK: scf.parallel
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c64_1 step %c32_3
// CHECK: scf.for{{.*}}%c0_6 to %c64_1 step %c32_3
// CHECK: scf.for{{.*}}%c0_6 to %c1024_5 step %c128_4
// CHECK: scf.for{{.*}}%c0_6 to %c128_4 step %c32_3
// CHECK: air.channel.put @channel_10
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK-NEXT: }
// CHECK: scf.parallel
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c512_2 step %c64_1
// CHECK: scf.for{{.*}}%c0_6 to %c64_1 step %c32_3
// CHECK: scf.for{{.*}}%c0_6 to %c64_1 step %c32_3
// CHECK: air.channel.get @channel_11
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK-NEXT: }
// CHECK: air.herd @herd_0
// CHECK: scf.for{{.*}}%c0_15 to %c512_14 step %c64_16
// CHECK: scf.for{{.*}}%c0_15 to %c512_14 step %c64_16
// CHECK: scf.for{{.*}}%c0_15 to %c64_16 step %c32_13
// CHECK: scf.for{{.*}}%c0_15 to %c64_16 step %c32_13
// CHECK: scf.for{{.*}}%c0_15 to %c1024_11 step %c128_12
// CHECK: scf.for{{.*}}%c0_15 to %c128_12 step %c32_13
// CHECK: air.channel.get @channel_10
// CHECK: }
// CHECK: }
// CHECK: air.channel.put @channel_11
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator

  func.func @legalize_memspace_sync(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>) {
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<1024x1024xi32>
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%alloc) : memref<1024x1024xi32>, memref<1024x1024xi32>, memref<1024x1024xi32> {
      air.segment @segment_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6, %arg14=%arg7, %arg15=%arg8, %arg16=%arg9) : index, index, index, index, memref<1024x1024xi32>, memref<1024x1024xi32>, memref<1024x1024xi32> {
        %c2_1 = arith.constant 2 : index
        %c2_2 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg17, %arg18) in (%arg19=%c2_1, %arg20=%c2_2) args(%arg21=%arg10, %arg22=%arg11, %arg23=%arg12, %arg24=%arg13, %arg25=%arg14, %arg26=%arg15, %arg27=%arg16) : index, index, index, index, memref<1024x1024xi32>, memref<1024x1024xi32>, memref<1024x1024xi32> {
          %c1_3 = arith.constant 1 : index
          %c1024 = arith.constant 1024 : index
          %c128 = arith.constant 128 : index
          %c32 = arith.constant 32 : index
          %c512 = arith.constant 512 : index
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = affine.apply #map()[%arg17]
          %1 = affine.apply #map()[%arg18]
          scf.for %arg28 = %c0 to %c512 step %c64 {
            scf.for %arg29 = %c0 to %c512 step %c64 {
              %2 = arith.addi %0, %arg28 : index
              %3 = arith.addi %1, %arg29 : index
              %alloc_4 = memref.alloc() : memref<64x64xi32, 1>
              scf.for %arg30 = %c0 to %c64 step %c32 {
                scf.for %arg31 = %c0 to %c64 step %c32 {
                  %4 = arith.addi %2, %arg30 : index
                  %5 = arith.addi %3, %arg31 : index
                  %alloc_5 = memref.alloc() : memref<32x32xi32, 2>
                  scf.for %arg32 = %c0 to %c1024 step %c128 {
                    %alloc_6 = memref.alloc() : memref<32x128xi32, 1>
                    air.dma_memcpy_nd (%alloc_6[] [] [], %arg25[%4, %arg32] [%c32, %c128] [%c1024, %c1_3]) {id = 1 : i32} : (memref<32x128xi32, 1>, memref<1024x1024xi32>)
                    scf.for %arg33 = %c0 to %c128 step %c32 {
                      %alloc_8 = memref.alloc() : memref<32x32xi32, 2>
                      air.dma_memcpy_nd (%alloc_8[] [] [], %alloc_6[%c0, %arg33] [%c32, %c32] [%c128, %c1_3]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<32x128xi32, 1>)
                      memref.dealloc %alloc_8 : memref<32x32xi32, 2>
                    }
                    memref.dealloc %alloc_6 : memref<32x128xi32, 1>
                  }
                  air.dma_memcpy_nd (%alloc_4[%arg30, %arg31] [%c32, %c32] [%c64, %c1_3], %alloc_5[] [] []) {id = 5 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
                  memref.dealloc %alloc_5 : memref<32x32xi32, 2>
                }
              }
              air.dma_memcpy_nd (%arg27[%2, %3] [%c64, %c64] [%c1024, %c1_3], %alloc_4[] [] []) {id = 6 : i32} : (memref<1024x1024xi32>, memref<64x64xi32, 1>)
              memref.dealloc %alloc_4 : memref<64x64xi32, 1>
            }
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    memref.copy %alloc, %arg2 : memref<1024x1024xi32> to memref<1024x1024xi32>
    return
  }
}
