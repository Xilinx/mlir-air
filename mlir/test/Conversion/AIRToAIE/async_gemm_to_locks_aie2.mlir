//===- async_gemm_to_locks.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-place-herds='num-rows=2 num-cols=2 row-anchor=3 col-anchor=5' -air-to-aie="emit-while-loop=false use-objectfifo=false row-offset=3 col-offset=5 device=xcve2802" %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(3, 0)
// CHECK:   %[[VAL_2:.*]] = AIE.tile(5, 1)
// CHECK:   %[[VAL_3:.*]] = AIE.tile(6, 1)
// CHECK:   %[[VAL_4:.*]] = AIE.tile(5, 3)
// CHECK:   %[[VAL_5:.*]] = AIE.tile(6, 3)
// CHECK:   %[[VAL_6:.*]] = AIE.tile(5, 4)
// CHECK:   %[[VAL_7:.*]] = AIE.tile(6, 4)
// CHECK-COUNT-46:    AIE.lock
// CHECK:   AIE.buffer(%[[VAL_2]]){{.*}}memref<64x64xi32, 1>
// CHECK:   AIE.buffer(%[[VAL_3]]){{.*}}memref<64x64xi32, 1>
// CHECK:   AIE.buffer(%[[VAL_2]]){{.*}}memref<64x64xi32, 1>
// CHECK:   AIE.buffer(%[[VAL_7]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_7]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_7]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_6]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_6]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_6]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_5]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_5]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_5]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_4]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_4]]){{.*}}memref<32x32xi32, 2>
// CHECK:   AIE.buffer(%[[VAL_4]]){{.*}}memref<32x32xi32, 2>
// CHECK:   %[[VAL_13:.*]] = AIE.mem(%[[VAL_7]]) {
// CHECK:   %[[VAL_14:.*]] = AIE.core(%[[VAL_7]]) {
// CHECK:   %[[VAL_15:.*]] = AIE.mem(%[[VAL_6]]) {
// CHECK:   %[[VAL_16:.*]] = AIE.core(%[[VAL_6]]) {
// CHECK:   %[[VAL_17:.*]] = AIE.mem(%[[VAL_5]]) {
// CHECK:   %[[VAL_18:.*]] = AIE.core(%[[VAL_5]]) {
// CHECK:   %[[VAL_19:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:   %[[VAL_20:.*]] = AIE.core(%[[VAL_4]]) {


#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_7 [1, 1]
  air.channel @channel_6 [2, 2]
  air.channel @channel_5 [2, 2]
  air.channel @channel_4 [2, 2]
  air.channel @channel_3 [2, 2]
  air.channel @channel_2 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @matmul(%arg0: memref<128x128xi32>, %arg1: memref<128x128xi32>, %arg2: memref<128x128xi32>) {
    %c2 = arith.constant 2 : index
    %async_token, %results = air.execute -> (memref<128x128xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xi32>
      air.execute_terminator %alloc : memref<128x128xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      memref.copy %arg2, %results : memref<128x128xi32> to memref<128x128xi32>
    }
    %0 = air.launch async [%async_token_0] (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%results) : memref<128x128xi32>, memref<128x128xi32>, memref<128x128xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %async_token_1, %results_2 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %1 = scf.for %arg10 = %c0 to %c128 step %c64 iter_args(%arg11 = %async_token_1) -> (!air.async.token) {
        %8 = air.channel.put async [%arg11]  @channel_0[] (%arg7[%results_2, %arg10] [%c64, %c64] [%c128, %c1]) {id = 1 : i32} : (memref<128x128xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %2 = scf.for %arg10 = %c0 to %c128 step %c64 iter_args(%arg11 = %async_token_3) -> (!air.async.token) {
        %8 = air.channel.put async [%arg11]  @channel_1[] (%arg8[%arg10, %results_4] [%c64, %c64] [%c128, %c1]) {id = 2 : i32} : (memref<128x128xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_5, %results_6 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_7, %results_8 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %3 = air.wait_all async [%async_token_7, %async_token_5] 
      %4 = scf.for %arg10 = %c0 to %c128 step %c64 iter_args(%arg11 = %3) -> (!air.async.token) {
        %8 = air.channel.put async [%arg11]  @channel_2[] (%arg9[%results_6, %results_8] [%c64, %c64] [%c128, %c1]) {id = 3 : i32} : (memref<128x128xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_9, %results_10 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_11, %results_12 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %5 = air.wait_all async [%async_token_11, %async_token_9] 
      %6 = scf.for %arg10 = %c0 to %c128 step %c64 iter_args(%arg11 = %5) -> (!air.async.token) {
        %8 = air.channel.get async [%arg11]  @channel_7[] (%arg9[%results_10, %results_12] [%c64, %c64] [%c128, %c1]) {id = 4 : i32} : (memref<128x128xi32>)
        scf.yield %8 : !air.async.token
      }
      %7 = air.segment async  args(%arg10=%arg3, %arg11=%arg4) : index, index attributes {id = 2 : i32} {
        %c32 = arith.constant 32 : index
        %c1_13 = arith.constant 1 : index
        %c2_14 = arith.constant 2 : index
        %c0_15 = arith.constant 0 : index
        %c128_16 = arith.constant 128 : index
        %c64_17 = arith.constant 64 : index
        %8 = air.wait_all async 
        %9 = scf.for %arg12 = %c0_15 to %c128_16 step %c64_17 iter_args(%arg13 = %8) -> (!air.async.token) {
          %async_token_18, %results_19 = air.execute -> (memref<64x64xi32, 1>) {
            %alloc = memref.alloc() : memref<64x64xi32, 1>
            air.execute_terminator %alloc : memref<64x64xi32, 1>
          }
          %async_token_20, %results_21 = air.execute -> (memref<64x64xi32, 1>) {
            %alloc = memref.alloc() : memref<64x64xi32, 1>
            air.execute_terminator %alloc : memref<64x64xi32, 1>
          }
          %async_token_22, %results_23 = air.execute -> (memref<64x64xi32, 1>) {
            %alloc = memref.alloc() : memref<64x64xi32, 1>
            air.execute_terminator %alloc : memref<64x64xi32, 1>
          }
          %10 = air.channel.get async [%async_token_18, %arg13]  @channel_0[] (%results_19[] [] []) {id = 5 : i32} : (memref<64x64xi32, 1>)
          %11 = air.channel.get async [%async_token_20, %arg13]  @channel_1[] (%results_21[] [] []) {id = 6 : i32} : (memref<64x64xi32, 1>)
          %12 = air.channel.get async [%async_token_22, %arg13]  @channel_2[] (%results_23[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>)
          %13 = scf.parallel (%arg14, %arg15) = (%c0_15, %c0_15) to (%c2_14, %c2_14) step (%c1_13, %c1_13) init (%10) -> !air.async.token {
            %async_token_27, %results_28 = air.execute -> (index) {
              %22 = affine.apply #map1()[%arg14]
              air.execute_terminator %22 : index
            }
            %20 = air.wait_all async [%async_token_27, %10] 
            %21 = scf.for %arg16 = %c0_15 to %c64_17 step %c32 iter_args(%arg17 = %20) -> (!air.async.token) {
              %22 = air.channel.put async [%arg17]  @channel_3[%arg14, %arg15] (%results_19[%results_28, %arg16] [%c32, %c32] [%c64_17, %c1_13]) {id = 8 : i32} : (memref<64x64xi32, 1>)
              scf.yield %22 : !air.async.token
            }
            scf.reduce(%21)  : !air.async.token {
            ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
              %22 = air.wait_all async [%arg16, %arg17] 
              scf.reduce.return %22 : !air.async.token
            }
            scf.yield
          }
          %14 = scf.parallel (%arg14, %arg15) = (%c0_15, %c0_15) to (%c2_14, %c2_14) step (%c1_13, %c1_13) init (%11) -> !air.async.token {
            %async_token_27, %results_28 = air.execute -> (index) {
              %22 = affine.apply #map1()[%arg15]
              air.execute_terminator %22 : index
            }
            %20 = air.wait_all async [%async_token_27, %11] 
            %21 = scf.for %arg16 = %c0_15 to %c64_17 step %c32 iter_args(%arg17 = %20) -> (!air.async.token) {
              %22 = air.channel.put async [%arg17]  @channel_4[%arg14, %arg15] (%results_21[%arg16, %results_28] [%c32, %c32] [%c64_17, %c1_13]) {id = 9 : i32} : (memref<64x64xi32, 1>)
              scf.yield %22 : !air.async.token
            }
            scf.reduce(%21)  : !air.async.token {
            ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
              %22 = air.wait_all async [%arg16, %arg17] 
              scf.reduce.return %22 : !air.async.token
            }
            scf.yield
          }
          %15 = scf.parallel (%arg14, %arg15) = (%c0_15, %c0_15) to (%c2_14, %c2_14) step (%c1_13, %c1_13) init (%12) -> !air.async.token {
            %async_token_27, %results_28 = air.execute -> (index) {
              %22 = affine.apply #map1()[%arg14]
              air.execute_terminator %22 : index
            }
            %async_token_29, %results_30 = air.execute -> (index) {
              %22 = affine.apply #map1()[%arg15]
              air.execute_terminator %22 : index
            }
            %20 = air.wait_all async [%async_token_29, %async_token_27, %12] 
            %21 = scf.for %arg16 = %c0_15 to %c64_17 step %c32 iter_args(%arg17 = %20) -> (!air.async.token) {
              %22 = air.channel.put async [%arg17]  @channel_5[%arg14, %arg15] (%results_23[%results_28, %results_30] [%c32, %c32] [%c64_17, %c1_13]) {id = 10 : i32} : (memref<64x64xi32, 1>)
              scf.yield %22 : !air.async.token
            }
            scf.reduce(%21)  : !air.async.token {
            ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
              %22 = air.wait_all async [%arg16, %arg17] 
              scf.reduce.return %22 : !air.async.token
            }
            scf.yield
          }
          %16 = scf.parallel (%arg14, %arg15) = (%c0_15, %c0_15) to (%c2_14, %c2_14) step (%c1_13, %c1_13) init (%12) -> !air.async.token {
            %async_token_27, %results_28 = air.execute -> (index) {
              %22 = affine.apply #map1()[%arg14]
              air.execute_terminator %22 : index
            }
            %async_token_29, %results_30 = air.execute -> (index) {
              %22 = affine.apply #map1()[%arg15]
              air.execute_terminator %22 : index
            }
            %20 = air.wait_all async [%async_token_29, %async_token_27, %12] 
            %21 = scf.for %arg16 = %c0_15 to %c64_17 step %c32 iter_args(%arg17 = %20) -> (!air.async.token) {
              %22 = air.channel.get async [%arg17]  @channel_6[%arg14, %arg15] (%results_23[%results_28, %results_30] [%c32, %c32] [%c64_17, %c1_13]) {id = 11 : i32} : (memref<64x64xi32, 1>)
              scf.yield %22 : !air.async.token
            }
            scf.reduce(%21)  : !air.async.token {
            ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
              %22 = air.wait_all async [%arg16, %arg17] 
              scf.reduce.return %22 : !air.async.token
            }
            scf.yield
          }
          %17 = air.herd @herd_0 async [%12, %11, %10]  tile (%arg14, %arg15) in (%arg16=%c2_14, %arg17=%c2_14) attributes {id = 3 : i32} {
            %c0_27 = arith.constant 0 : index
            %c64_28 = arith.constant 64 : index
            %c32_29 = arith.constant 32 : index
            %20 = air.wait_all async 
            %21 = scf.for %arg18 = %c0_27 to %c64_28 step %c32_29 iter_args(%arg19 = %20) -> (!air.async.token) {
              %async_token_30, %results_31 = air.execute -> (memref<32x32xi32, 2>) {
                %alloc = memref.alloc() : memref<32x32xi32, 2>
                air.execute_terminator %alloc : memref<32x32xi32, 2>
              }
              %async_token_32, %results_33 = air.execute -> (memref<32x32xi32, 2>) {
                %alloc = memref.alloc() : memref<32x32xi32, 2>
                air.execute_terminator %alloc : memref<32x32xi32, 2>
              }
              %async_token_34, %results_35 = air.execute -> (memref<32x32xi32, 2>) {
                %alloc = memref.alloc() : memref<32x32xi32, 2>
                air.execute_terminator %alloc : memref<32x32xi32, 2>
              }
              %22 = air.channel.get async [%async_token_30, %arg19]  @channel_3[%arg14, %arg15] (%results_31[] [] []) {id = 12 : i32} : (memref<32x32xi32, 2>)
              %23 = air.channel.get async [%async_token_32, %arg19]  @channel_4[%arg14, %arg15] (%results_33[] [] []) {id = 13 : i32} : (memref<32x32xi32, 2>)
              %24 = air.channel.get async [%async_token_34, %arg19]  @channel_5[%arg14, %arg15] (%results_35[] [] []) {id = 14 : i32} : (memref<32x32xi32, 2>)
              %async_token_36 = air.execute [%24, %23, %22] {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_31, %results_33 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_35 : memref<32x32xi32, 2>)
              }
              %25 = air.channel.put async [%async_token_36]  @channel_6[%arg14, %arg15] (%results_35[] [] []) {id = 15 : i32} : (memref<32x32xi32, 2>)
              %async_token_37 = air.execute [%async_token_36] {
                memref.dealloc %results_31 : memref<32x32xi32, 2>
              }
              %async_token_38 = air.execute [%async_token_36] {
                memref.dealloc %results_33 : memref<32x32xi32, 2>
              }
              %async_token_39 = air.execute [%25] {
                memref.dealloc %results_35 : memref<32x32xi32, 2>
              }
              scf.yield %25 : !air.async.token
            }
            air.herd_terminator
          }
          %18 = air.channel.put async [%17]  @channel_7[] (%results_23[] [] []) {id = 16 : i32} : (memref<64x64xi32, 1>)
          %async_token_24 = air.execute [%10] {
            memref.dealloc %results_19 : memref<64x64xi32, 1>
          }
          %async_token_25 = air.execute [%11] {
            memref.dealloc %results_21 : memref<64x64xi32, 1>
          }
          %async_token_26 = air.execute [%18] {
            memref.dealloc %results_23 : memref<64x64xi32, 1>
          }
          %19 = air.wait_all async [%16, %15, %14, %18, %13] 
          scf.yield %19 : !air.async.token
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
