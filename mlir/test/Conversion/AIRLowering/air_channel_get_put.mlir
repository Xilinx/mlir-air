//===- air_channel_get_put.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --split-input-file -air-to-std | FileCheck %s

// CHECK-LABEL:   func.func @single_put_get
// CHECK: affine.for %{{.*}} 0 to 2
// CHECK: affine.for %{{.*}} 0 to 2
// CHECK: airrt.segment_load
// CHECK: airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
module {
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @single_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c2_0 = arith.constant 2 : index
    air.launch (%arg2, %arg3) in (%arg4=%c2_0, %arg5=%c2_0) args(%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.channel.put async  @channel_0[%c0, %c0] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 1 : i32} : (memref<32x16xi32>)
      %1 = air.channel.get async  @channel_1[%c0, %c0] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 2 : i32} : (memref<32x16xi32>)
      air.segment @segment_0 {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) {
          %c0_4 = arith.constant 0 : index
          %c1_4 = arith.constant 1 : index
          %c32_5 = arith.constant 32 : index
          %c16_6 = arith.constant 16 : index
          %c8_7 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_8 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_0[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c1_4]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %2 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %2, %alloc_8[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          air.channel.put  @channel_1[%arg10, %arg11] (%alloc_8[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c1_4]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          memref.dealloc %alloc_8 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @par_put_get
// CHECK: affine.for %{{.*}} 0 to 1
// CHECK: affine.for %{{.*}} 0 to 1
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.herd_load "herd_0" () : () -> i64

module {
  air.channel @channel_3 [2, 2]
  air.channel @channel_2 [2, 2]
  func.func @par_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c1_0 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_0) args(%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.wait_all async 
      %1 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = air.channel.put async  @channel_2[%a2, %a3] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 1 : i32} : (memref<32x16xi32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      %2 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = air.channel.get async  @channel_3[%a2, %a3] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 2 : i32} : (memref<32x16xi32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      air.segment @segment_0  args(%arg6=%arg2, %arg7=%arg3, %arg8=%arg4, %arg9=%arg5) : index, index, index, index {
        %c2_2 = arith.constant 2 : index
        %c2_3 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c2_2, %arg13=%c2_3) args(%arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index {
          %c0_4 = arith.constant 0 : index
          %c1_4 = arith.constant 1 : index
          %c32_5 = arith.constant 32 : index
          %c16_6 = arith.constant 16 : index
          %c8_7 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_8 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_2[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c1_4]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %3 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %3, %alloc_8[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          air.channel.put  @channel_3[%arg10, %arg11] (%alloc_8[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c1_4]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          memref.dealloc %alloc_8 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @par_with_for_put_get
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: affine.for %{{.*}} 0 to 2
// CHECK:   affine.for %{{.*}} 0 to 2
// CHECK:     airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:     scf.for
// CHECK:       airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:       scf.yield
// CHECK: airrt.herd_load "herd_0" () : () -> i64
module {
  air.channel @channel_5 [2, 2]
  air.channel @channel_4 [2, 2]
  func.func @par_with_for_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_1) args (%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.wait_all async 
      %1 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = air.channel.put async  @channel_4[%a2, %a3] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 1 : i32} : (memref<32x16xi32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      %2 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = scf.for %a4 = %c0 to %c2 step %c1 iter_args(%a5 = %0) -> (!air.async.token) {
          %4 = air.channel.get async [%a5]  @channel_5[%a2, %a3] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c1]) {id = 2 : i32} : (memref<32x16xi32>)
          scf.yield %4 : !air.async.token
        }
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      air.segment @segment_0  args(%arg6=%arg2, %arg7=%arg3, %arg8=%arg4, %arg9=%arg5) : index, index, index, index {
        %c2_2 = arith.constant 2 : index
        %c2_3 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c2_2, %arg13=%c2_3) args(%arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index {
          %c0_4 = arith.constant 0 : index
          %c2_5 = arith.constant 2 : index
          %c1_6 = arith.constant 1 : index
          %c32_7 = arith.constant 32 : index
          %c16_8 = arith.constant 16 : index
          %c8_9 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_10 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_4[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_9, %c16_8] [%c32_7, %c1_6]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %3 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %3, %alloc_10[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          scf.for %arg18 = %c0_4 to %c2_5 step %c1_6 {
            air.channel.put  @channel_5[%arg10, %arg11] (%alloc_10[%c0_4, %c0_4] [%c8_9, %c16_8] [%c32_7, %c1_6]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          }
          memref.dealloc %alloc_10 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
        }
      }
    }
    return
  }
}
// -----

// CHECK-LABEL:   func.func @one_d_scf_parallel
// CHECK: affine.for
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<128xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
// CHECK: airrt.herd_load "herd_0" () : () -> i64

#map = affine_map<(d0)[] -> (d0 * 64)>
module {
  air.channel @channel_6 [1, 1]
  func.func @one_d_scf_parallel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg2) in (%arg3=%c2) args(%arg4=%arg0) : memref<128xf32> attributes {id = 1 : i32} {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map(%arg2)[]
        air.execute_terminator %3 : index
      }
      %1 = air.channel.put async [%async_token]  @channel_6[] (%arg4[%results] [%c64] [%c1]) {id = 1 : i32} : (memref<128xf32>)
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c1_0 = arith.constant 1 : index
        %c2_1 = arith.constant 2 : index
        %3 = air.wait_all async 
        %async_token_2, %results_3 = air.execute -> (memref<64xf32, 1>) {
          %alloc = memref.alloc() : memref<64xf32, 1>
          air.execute_terminator %alloc : memref<64xf32, 1>
        }
        %4 = air.channel.get async [%3, %async_token_2]  @channel_6[] (%results_3[] [] []) {id = 3 : i32} : (memref<64xf32, 1>)
        %5 = air.herd @herd_0 async [%4]  tile (%arg5, %arg6) in (%arg7=%c1_0, %arg8=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token_5, %results_6 = air.execute -> (memref<32xf32, 2>) {
            %alloc = memref.alloc() : memref<32xf32, 2>
            air.execute_terminator %alloc : memref<32xf32, 2>
          }
          %async_token_7 = air.execute [%async_token_5] {
            memref.dealloc %results_6 : memref<32xf32, 2>
          }
        }
        %async_token_4 = air.execute [%4] {
          memref.dealloc %results_3 : memref<64xf32, 1>
        }
      }
    }
    return
  }
}

// -----

// Check for the generation of a blocking airrt.wait_all at the end of an air.launch.

// CHECK-LABEL:   func.func @air_dep_in_launch
// CHECK: affine.for
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: %[[TOKEN1:.*]] = airrt.dma_memcpy_nd
// CHECK: %[[WAIT1:.*]] = airrt.wait_all %[[TOKEN1]]
// CHECK: %[[TOKEN2:.*]] = airrt.dma_memcpy_nd
// CHECK: %[[WAIT2:.*]] = airrt.wait_all %[[TOKEN2]]
// CHECK: %[[TOKEN3:.*]] = airrt.dma_memcpy_nd
// CHECK: airrt.wait_all %[[TOKEN3]], %[[WAIT1]], %[[WAIT2]]

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  func.func @air_dep_in_launch(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>, %arg2: memref<512xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async () in () args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<512xbf16>, memref<512xbf16>, memref<512xbf16> {
      %1 = air.channel.put async  @channel_0[] (%arg7[] [] []) {metadata = @airMemcpyId13} : (memref<512xbf16>)
      %5 = air.channel.put async  @channel_1[] (%arg8[] [] []) {metadata = @airMemcpyId17} : (memref<512xbf16>)
      %9 = air.channel.get async [%1, %5]  @channel_2[] (%arg9[] [] []) {metadata = @airMemcpyId94} : (memref<512xbf16>)
      %13 = air.segment @segment_0 async {
        %async_token_16, %results_17 = air.execute -> (memref<512xbf16, 1>) {
          %alloc = memref.alloc() : memref<512xbf16, 1>
          air.execute_terminator %alloc : memref<512xbf16, 1>
        }
        %async_token_30, %results_31 = air.execute -> (memref<512xbf16, 1>) {
          %alloc = memref.alloc() : memref<512xbf16, 1>
          air.execute_terminator %alloc : memref<512xbf16, 1>
        }
        %async_token_38, %results_39 = air.execute -> (memref<512xbf16, 1>) {
          %alloc = memref.alloc() : memref<512xbf16, 1>
          air.execute_terminator %alloc : memref<512xbf16, 1>
        }
        %14 = air.channel.get async [%async_token_38]  @channel_0[] (%results_39[] [] []) {id = 13 : i32} : (memref<512xbf16, 1>)
        %18 = air.channel.get async [%async_token_30]  @channel_1[] (%results_31[] [] []) {id = 17 : i32} : (memref<512xbf16, 1>)
        %71 = air.channel.put async [%14, %18]  @channel_2[] (%results_17[] [] []) {id = 94 : i32} : (memref<512xbf16, 1>)
        %async_token_40 = air.execute [%14] {
          memref.dealloc %results_39 : memref<512xbf16, 1>
        }
        %async_token_44 = air.execute [%18] {
          memref.dealloc %results_31 : memref<512xbf16, 1>
        }
        %async_token_51 = air.execute [%71] {
          memref.dealloc %results_17 : memref<512xbf16, 1>
        }
      }
      %15 = air.wait_all async [%1, %5, %9]
    }
    return
  }
}

// -----

// Specialize metadata array.

// CHECK-LABEL:   func.func @metadataArray
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_0_0}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_0_1}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_0_2}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_0_3}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_1_0}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_1_1}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_1_2}
// CHECK: airrt.dma_memcpy_nd{{.*}}{metadata = @air_channel_1_3}

module {
  aie.device(npu1_4col) {
    aie.shim_dma_allocation @air_channel_1_0(S2MM, 0, 0)
    memref.global "public" @air_channel_1_0 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_1_1(S2MM, 1, 0)
    memref.global "public" @air_channel_1_1 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_1_2(S2MM, 0, 1)
    memref.global "public" @air_channel_1_2 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_1_3(S2MM, 1, 1)
    memref.global "public" @air_channel_1_3 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_0_0(MM2S, 0, 0)
    memref.global "public" @air_channel_0_0 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_0_1(MM2S, 1, 0)
    memref.global "public" @air_channel_0_1 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_0_2(MM2S, 0, 1)
    memref.global "public" @air_channel_0_2 : memref<4x2xf32, 2 : i32>
    aie.shim_dma_allocation @air_channel_0_3(MM2S, 1, 1)
    memref.global "public" @air_channel_0_3 : memref<4x2xf32, 2 : i32>
  } {sym_name = "herd_0"}
  air.channel @channel_0 [2, 2]
  air.channel @channel_1 [2, 2]
  func.func @metadataArray(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %0 = air.launch async () in () args(%arg8=%arg0, %arg9=%arg1) : memref<*xf32>, memref<*xf32> attributes {dummyLaunch = true, id = 1 : i32} {
      %c16448 = arith.constant 16448 : index
      %c16384 = arith.constant 16384 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %1 = air.channel.put async  @channel_0[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 1 : i32, metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<*xf32>)
      %2 = air.channel.put async  @channel_0[%c0, %c1] (%arg8[%c0, %c0, %c0, %c64] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 1 : i32, metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<*xf32>)
      %3 = air.channel.put async  @channel_0[%c1, %c0] (%arg8[%c0, %c0, %c0, %c16384] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 1 : i32, metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<*xf32>)
      %4 = air.channel.put async  @channel_0[%c1, %c1] (%arg8[%c0, %c0, %c0, %c16448] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 1 : i32, metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<*xf32>)
      %5 = air.channel.get async  @channel_1[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 2 : i32, metadataArray = [{base = "air_channel_1_0", index = 0 : i32}, {base = "air_channel_1_1", index = 1 : i32}, {base = "air_channel_1_2", index = 2 : i32}, {base = "air_channel_1_3", index = 3 : i32}]} : (memref<*xf32>)
      %6 = air.channel.get async  @channel_1[%c0, %c1] (%arg9[%c0, %c0, %c0, %c64] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 2 : i32, metadataArray = [{base = "air_channel_1_0", index = 0 : i32}, {base = "air_channel_1_1", index = 1 : i32}, {base = "air_channel_1_2", index = 2 : i32}, {base = "air_channel_1_3", index = 3 : i32}]} : (memref<*xf32>)
      %7 = air.channel.get async  @channel_1[%c1, %c0] (%arg9[%c0, %c0, %c0, %c16384] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 2 : i32, metadataArray = [{base = "air_channel_1_0", index = 0 : i32}, {base = "air_channel_1_1", index = 1 : i32}, {base = "air_channel_1_2", index = 2 : i32}, {base = "air_channel_1_3", index = 3 : i32}]} : (memref<*xf32>)
      %8 = air.channel.get async  @channel_1[%c1, %c1] (%arg9[%c0, %c0, %c0, %c16448] [%c32, %c32, %c4, %c2] [%c512, %c2, %c128, %c1]) {id = 2 : i32, metadataArray = [{base = "air_channel_1_0", index = 0 : i32}, {base = "air_channel_1_1", index = 1 : i32}, {base = "air_channel_1_2", index = 2 : i32}, {base = "air_channel_1_3", index = 3 : i32}]} : (memref<*xf32>)
      %9 = air.herd @herd_0 async  tile (%arg10, %arg11) in (%arg12=%c1, %arg13=%c4) attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_i32 = arith.constant 1 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c2_0 = arith.constant 2 : index
        %10 = arith.remsi %arg11, %c2_0 : index
        %11 = arith.divsi %arg11, %c2_0 : index
        %12 = air.wait_all async 
        %13 = scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg15 = %12) -> (!air.async.token)  : i32 {
          %14 = scf.for %arg16 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg17 = %arg15) -> (!air.async.token)  : i32 {
            %async_token, %results = air.execute -> (memref<4x2xf32, 2 : i32>) {
              %alloc = memref.alloc() : memref<4x2xf32, 2 : i32>
              air.execute_terminator %alloc : memref<4x2xf32, 2 : i32>
            }
            %15 = air.channel.get async [%arg17, %async_token]  @channel_0[%11, %10] (%results[] [] []) {id = 3 : i32} : (memref<4x2xf32, 2 : i32>)
            %16 = air.channel.put async [%15]  @channel_1[%11, %10] (%results[] [] []) {id = 4 : i32} : (memref<4x2xf32, 2 : i32>)
            scf.yield %16 : !air.async.token
          }
          scf.yield %14 : !air.async.token
        }
      }
      air.wait_all [%9] 
    }
    return
  }
}
