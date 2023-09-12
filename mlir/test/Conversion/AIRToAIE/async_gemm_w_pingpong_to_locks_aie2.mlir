//===- async_gemm_w_ping_pong_to_locks_aie2.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-aie="emit-while-loop=false use-objectfifo=false row-offset=3 col-offset=5 device=xcve2802" %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2802) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(3, 0)
// CHECK:   %[[VAL_2:.*]] = AIE.tile(5, 1)
// CHECK:   %[[VAL_3:.*]] = AIE.tile(6, 1)
// CHECK:   %[[VAL_4:.*]] = AIE.tile(5, 3)
// CHECK:   %[[VAL_5:.*]] = AIE.tile(6, 3)
// CHECK:   %[[VAL_6:.*]] = AIE.tile(5, 4)
// CHECK:   %[[VAL_7:.*]] = AIE.tile(6, 4)
// CHECK-COUNT-12:    AIE.lock(%[[VAL_3]], {{.*}})
// CHECK-COUNT-28:    AIE.lock(%[[VAL_2]], {{.*}})
// CHECK-COUNT-12:    AIE.lock(%[[VAL_4]], {{.*}})
// CHECK-COUNT-12:    AIE.lock(%[[VAL_5]], {{.*}})
// CHECK-COUNT-12:    AIE.lock(%[[VAL_6]], {{.*}})
// CHECK-COUNT-12:    AIE.lock(%[[VAL_7]], {{.*}})
// CHECK:    AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<64x64xi32, 1>
// CHECK:    AIE.buffer(%[[VAL_3]]) {sym_name = {{.*}}} : memref<64x128xi32, 1>
// CHECK:    AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<128x64xi32, 1>
// CHECK:    AIE.buffer(%[[VAL_3]]) {sym_name = {{.*}}} : memref<64x128xi32, 1>
// CHECK:    AIE.buffer(%[[VAL_2]]) {sym_name = {{.*}}} : memref<128x64xi32, 1>
// CHECK-COUNT-20:    AIE.buffer({{.*}}) {sym_name = {{.*}}} : memref<32x32xi32, 2>
// CHECK:   AIE.mem(%[[VAL_7]])
// CHECK:   AIE.core(%[[VAL_7]]) {
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     scf.for
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:     }
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:   } {elf_file = 
// CHECK:   AIE.mem(%[[VAL_6]])
// CHECK:   AIE.core(%[[VAL_6]])
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     scf.for
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:     }
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:   } {elf_file = 
// CHECK:   AIE.mem(%[[VAL_5]])
// CHECK:   AIE.core(%[[VAL_5]])
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     scf.for
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:     }
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:   } {elf_file = 
// CHECK:   AIE.mem(%[[VAL_4]])
// CHECK:   AIE.core(%[[VAL_4]])
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     scf.for
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:       linalg.matmul
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:       AIE.useLock({{.*}}, Release, 1)
// CHECK:     }
// CHECK:     AIE.useLock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:     AIE.useLock({{.*}}, Release, 1)
// CHECK:   } {elf_file = 

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
module {
  air.channel @channel_9 [1, 1]
  air.channel @channel_8 [2, 2]
  air.channel @channel_7 [2, 2]
  air.channel @channel_6 [1, 1]
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func @matmul(%arg0: memref<128x512xi32>, %arg1: memref<512x128xi32>, %arg2: memref<128x128xi32>) {
    %c2 = arith.constant 2 : index
    %async_token, %results = air.execute -> (memref<128x128xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xi32>
      air.execute_terminator %alloc : memref<128x128xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      memref.copy %arg2, %results : memref<128x128xi32> to memref<128x128xi32>
    }
    %0 = air.launch async [%async_token_0] (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%results) : memref<128x512xi32>, memref<512x128xi32>, memref<128x128xi32> attributes {id = 1 : i32} {
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %async_token_1, %results_2 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %1 = air.channel.put async [%async_token_3, %async_token_1]  @channel_4[] (%arg9[%results_2, %results_4] [%c64, %c64] [%c128, %c1]) {id = 1 : i32} : (memref<128x128xi32>)
      %async_token_5, %results_6 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %2 = air.wait_all async 
      %3 = scf.for %arg10 = %c0 to %c512 step %c128 iter_args(%arg11 = %2) -> (!air.async.token) {
        %8 = air.channel.put async [%arg11, %async_token_5]  @channel_5[] (%arg7[%results_6, %arg10] [%c64, %c128] [%c512, %c1]) {id = 2 : i32} : (memref<128x512xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_7, %results_8 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg4]
        air.execute_terminator %8 : index
      }
      %4 = air.wait_all async 
      %5 = scf.for %arg10 = %c0 to %c512 step %c128 iter_args(%arg11 = %4) -> (!air.async.token) {
        %8 = air.channel.put async [%arg11, %async_token_7]  @channel_6[] (%arg8[%arg10, %results_8] [%c128, %c64] [%c128, %c1]) {id = 3 : i32} : (memref<512x128xi32>)
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
      %6 = air.channel.get async [%async_token_11, %async_token_9]  @channel_9[] (%arg9[%results_10, %results_12] [%c64, %c64] [%c128, %c1]) {id = 4 : i32} : (memref<128x128xi32>)
      %7 = air.segment async  attributes {id = 2 : i32, x_loc = 5 : i64, x_size = 2 : i64, y_loc = 3 : i64, y_size = 2 : i64} {
        %c256 = arith.constant 256 : index
        %c32 = arith.constant 32 : index
        %c1_13 = arith.constant 1 : index
        %c64_14 = arith.constant 64 : index
        %c2_15 = arith.constant 2 : index
        %c0_16 = arith.constant 0 : index
        %c512_17 = arith.constant 512 : index
        %c128_18 = arith.constant 128 : index
        %8 = air.wait_all async 
        %async_token_19, %results_20 = air.execute -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %9 = air.channel.get async [%async_token_19, %8]  @channel_4[] (%results_20[] [] []) {id = 5 : i32} : (memref<64x64xi32, 1>)
        %10 = scf.for %arg10 = %c0_16 to %c512_17 step %c128_18 iter_args(%arg11 = %9) -> (!air.async.token) {
          %13 = air.herd @herd_0 async [%arg11]  tile (%arg12, %arg13) in (%arg14=%c2_15, %arg15=%c2_15) attributes {id = 3 : i32, x_loc = 5 : i64, y_loc = 3 : i64} {
            %c64_30 = arith.constant 64 : index
            %c0_31 = arith.constant 0 : index
            %c128_32 = arith.constant 128 : index
            %17 = air.wait_all async 
            %async_token_33, %results_34 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %18 = air.channel.get async [%async_token_33, %17]  @channel_7[%arg12, %arg13] (%results_34[] [] []) {id = 14 : i32} : (memref<32x32xi32, 2>)
            %async_token_35, %results_36 = air.execute [%18] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_37, %results_38 = air.execute [%async_token_35] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_39, %results_40 = air.execute [%async_token_37] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_41, %results_42 = air.execute [%async_token_37] -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %19:4 = scf.for %arg16 = %c0_31 to %c128_32 step %c64_30 iter_args(%arg17 = %async_token_39, %arg18 = %async_token_41, %arg19 = %async_token_41, %arg20 = %async_token_41) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
              %21 = affine.if #set()[%arg12, %arg13] -> !air.async.token {
                %26 = air.channel.get async [%arg20, %async_token_39, %arg17]  @channel_0[%arg12, %arg13] (%results_40[] [] []) {id = 15 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              } else {
                %26 = air.channel.get async [%arg20, %async_token_39, %arg17]  @channel_1[%arg12, %arg13] (%results_40[] [] []) {id = 16 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              }
              %22 = affine.if #set1()[%arg12, %arg13] -> !air.async.token {
                %26 = air.channel.get async [%arg20, %async_token_41, %arg17]  @channel_2[%arg12, %arg13] (%results_42[] [] []) {id = 17 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              } else {
                %26 = air.channel.get async [%arg20, %async_token_41, %arg17]  @channel_3[%arg12, %arg13] (%results_42[] [] []) {id = 18 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              }
              %async_token_44 = air.execute [%arg19, %22, %21] {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_40, %results_42 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_34 : memref<32x32xi32, 2>)
              }
              %async_token_45 = air.execute {
                memref.dealloc %results_40 : memref<32x32xi32, 2>
              }
              %async_token_46 = air.execute {
                memref.dealloc %results_42 : memref<32x32xi32, 2>
              }
              %23 = affine.if #set()[%arg12, %arg13] -> !air.async.token {
                %26 = air.channel.get async [%22, %21, %arg18]  @channel_0[%arg12, %arg13] (%results_38[] [] []) {id = 15 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              } else {
                %26 = air.channel.get async [%22, %21, %arg18]  @channel_1[%arg12, %arg13] (%results_38[] [] []) {id = 16 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              }
              %24 = affine.if #set1()[%arg12, %arg13] -> !air.async.token {
                %26 = air.channel.get async [%22, %21, %arg18]  @channel_2[%arg12, %arg13] (%results_36[] [] []) {id = 17 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              } else {
                %26 = air.channel.get async [%22, %21, %arg18]  @channel_3[%arg12, %arg13] (%results_36[] [] []) {id = 18 : i32} : (memref<32x32xi32, 2>)
                affine.yield %26 : !air.async.token
              }
              %async_token_47 = air.execute [%async_token_44, %24, %23] {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_38, %results_36 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_34 : memref<32x32xi32, 2>)
              }
              %async_token_48 = air.execute {
                memref.dealloc %results_38 : memref<32x32xi32, 2>
              }
              %async_token_49 = air.execute {
                memref.dealloc %results_36 : memref<32x32xi32, 2>
              }
              %25 = air.wait_all async [%23, %24] 
              scf.yield %async_token_44, %async_token_47, %async_token_47, %25 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
            }
            %20 = air.channel.put async [%19#1]  @channel_8[%arg12, %arg13] (%results_34[] [] []) {id = 19 : i32} : (memref<32x32xi32, 2>)
            %async_token_43 = air.execute [%20] {
              memref.dealloc %results_34 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          %14 = scf.parallel (%arg12, %arg13) = (%c0_16, %c0_16) to (%c2_15, %c2_15) step (%c1_13, %c1_13) init (%arg11) -> !air.async.token {
            %async_token_30, %results_31 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg12]
              air.execute_terminator %18 : index
            }
            %async_token_32, %results_33 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg13]
              air.execute_terminator %18 : index
            }
            %17 = air.channel.put async [%async_token_32, %async_token_30, %arg11]  @channel_7[%arg12, %arg13] (%results_20[%results_31, %results_33] [%c32, %c32] [%c64_14, %c1_13]) {id = 12 : i32} : (memref<64x64xi32, 1>)
            scf.reduce(%17)  : !air.async.token {
            ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
              %18 = air.wait_all async [%arg14, %arg15] 
              scf.reduce.return %18 : !air.async.token
            }
            scf.yield
          }
          %15 = scf.parallel (%arg12, %arg13) = (%c0_16, %c0_16) to (%c2_15, %c2_15) step (%c1_13, %c1_13) init (%arg11) -> !air.async.token {
            %async_token_30, %results_31 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg12]
              air.execute_terminator %18 : index
            }
            %async_token_32, %results_33 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg13]
              air.execute_terminator %18 : index
            }
            %17 = air.channel.get async [%async_token_32, %async_token_30, %arg11]  @channel_8[%arg12, %arg13] (%results_20[%results_31, %results_33] [%c32, %c32] [%c64_14, %c1_13]) {id = 13 : i32} : (memref<64x64xi32, 1>)
            scf.reduce(%17)  : !air.async.token {
            ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
              %18 = air.wait_all async [%arg14, %arg15] 
              scf.reduce.return %18 : !air.async.token
            }
            scf.yield
          }
          %16 = air.wait_all async [%13, %14, %15] 
          scf.yield %16 : !air.async.token
        }
        %async_token_21, %results_22 = air.execute [%9] -> (memref<128x64xi32, 1>) {
          %alloc = memref.alloc() : memref<128x64xi32, 1>
          air.execute_terminator %alloc : memref<128x64xi32, 1>
        }
        %async_token_23, %results_24 = air.execute [%async_token_21] -> (memref<64x128xi32, 1>) {
          %alloc = memref.alloc() : memref<64x128xi32, 1>
          air.execute_terminator %alloc : memref<64x128xi32, 1>
        }
        %async_token_25, %results_26 = air.execute [%async_token_23] -> (memref<64x128xi32, 1>) {
          %alloc = memref.alloc() : memref<64x128xi32, 1>
          air.execute_terminator %alloc : memref<64x128xi32, 1>
        }
        %async_token_27, %results_28 = air.execute [%async_token_23] -> (memref<128x64xi32, 1>) {
          %alloc = memref.alloc() : memref<128x64xi32, 1>
          air.execute_terminator %alloc : memref<128x64xi32, 1>
        }
        %11:4 = scf.for %arg10 = %c0_16 to %c512_17 step %c256 iter_args(%arg11 = %async_token_25, %arg12 = %async_token_27, %arg13 = %async_token_27, %arg14 = %async_token_27) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %13 = air.channel.get async [%arg14, %async_token_25, %arg11]  @channel_5[] (%results_26[] [] []) {id = 6 : i32} : (memref<64x128xi32, 1>)
          %14 = air.channel.get async [%arg14, %async_token_27, %arg11]  @channel_6[] (%results_28[] [] []) {id = 7 : i32} : (memref<128x64xi32, 1>)
          %15 = air.wait_all async [%arg13, %13] 
          %16 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %15) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_0[] (%results_26[%c0_16, %arg15] [%c32, %c32] [%c128_18, %c1_13]) {id = 8 : i32} : (memref<64x128xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %17 = air.wait_all async [%arg13, %13] 
          %18 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %17) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_1[] (%results_26[%c32, %arg15] [%c32, %c32] [%c128_18, %c1_13]) {id = 9 : i32} : (memref<64x128xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %19 = air.wait_all async [%arg13, %14] 
          %20 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %19) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_2[] (%results_28[%arg15, %c0_16] [%c32, %c32] [%c64_14, %c1_13]) {id = 10 : i32} : (memref<128x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %21 = air.wait_all async [%arg13, %14] 
          %22 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %21) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_3[] (%results_28[%arg15, %c32] [%c32, %c32] [%c64_14, %c1_13]) {id = 11 : i32} : (memref<128x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %async_token_30 = air.execute {
            memref.dealloc %results_26 : memref<64x128xi32, 1>
          }
          %async_token_31 = air.execute {
            memref.dealloc %results_28 : memref<128x64xi32, 1>
          }
          %23 = air.wait_all async [%13, %14, %22, %20, %18, %16] 
          %24 = air.channel.get async [%14, %13, %arg12]  @channel_5[] (%results_24[] [] []) {id = 6 : i32} : (memref<64x128xi32, 1>)
          %25 = air.channel.get async [%14, %13, %arg12]  @channel_6[] (%results_22[] [] []) {id = 7 : i32} : (memref<128x64xi32, 1>)
          %26 = air.wait_all async [%23, %24] 
          %27 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %26) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_0[] (%results_24[%c0_16, %arg15] [%c32, %c32] [%c128_18, %c1_13]) {id = 8 : i32} : (memref<64x128xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %28 = air.wait_all async [%23, %24] 
          %29 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %28) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_1[] (%results_24[%c32, %arg15] [%c32, %c32] [%c128_18, %c1_13]) {id = 9 : i32} : (memref<64x128xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %30 = air.wait_all async [%23, %25] 
          %31 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %30) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_2[] (%results_22[%arg15, %c0_16] [%c32, %c32] [%c64_14, %c1_13]) {id = 10 : i32} : (memref<128x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %32 = air.wait_all async [%23, %25] 
          %33 = scf.for %arg15 = %c0_16 to %c128_18 step %c32 iter_args(%arg16 = %32) -> (!air.async.token) {
            %36 = air.channel.put async [%arg16]  @channel_3[] (%results_22[%arg15, %c32] [%c32, %c32] [%c64_14, %c1_13]) {id = 11 : i32} : (memref<128x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %async_token_32 = air.execute {
            memref.dealloc %results_24 : memref<64x128xi32, 1>
          }
          %async_token_33 = air.execute {
            memref.dealloc %results_22 : memref<128x64xi32, 1>
          }
          %34 = air.wait_all async [%24, %25, %33, %31, %29, %27] 
          %35 = air.wait_all async [%24, %25] 
          scf.yield %23, %34, %34, %35 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %12 = air.channel.put async [%10, %11#1]  @channel_9[] (%results_20[] [] []) {id = 20 : i32} : (memref<64x64xi32, 1>)
        %async_token_29 = air.execute [%12] {
          memref.dealloc %results_20 : memref<64x64xi32, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
