//===- multi_token_loop_race.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Race condition caused by for loop with multiple async tokens

// CHECK-COUNT-132: "name": "DmaMemcpyNdOp",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<()[s0] -> (s0 + 16)>
module {
  func.func @test(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %async_token, %results = air.execute -> (memref<512x512xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<512x512xbf16>
      air.execute_terminator %alloc : memref<512x512xbf16>
    }
    %async_token_0 = air.execute [%async_token] {
      memref.copy %arg2, %results : memref<512x512xbf16> to memref<512x512xbf16>
    }
    %0 = air.launch async [%async_token_0] (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%results) : memref<512x512xbf16>, memref<512x512xbf16>, memref<512x512xbf16> attributes {id = 1 : i32} {
      %1 = air.segment async  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<512x512xbf16>, memref<512x512xbf16>, memref<512x512xbf16> attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c1_1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %async_token_2, %results_3 = air.execute -> (index) {
          %6 = affine.apply #map()[%arg10]
          air.execute_terminator %6 : index
        }
        %async_token_4, %results_5 = air.execute -> (index) {
          %6 = affine.apply #map()[%arg11]
          air.execute_terminator %6 : index
        }
        %2 = air.wait_all async [%async_token_4, %async_token_2] 
        %async_token_6, %results_7 = air.execute -> (memref<64x64xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %alloc : memref<64x64xbf16, 1>
        }
        %3 = air.dma_memcpy_nd async [%async_token_6, %2] (%results_7[] [] [], %arg14[%results_3, %results_5] [%c64, %c64] [%c512, %c1_1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %4 = scf.for %arg15 = %c0 to %c512 step %c64 iter_args(%arg16 = %3) -> (!air.async.token) {
          %async_token_9, %results_10 = air.execute -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          }
          %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          }
          %6 = air.dma_memcpy_nd async [%async_token_9, %arg16] (%results_10[] [] [], %arg12[%results_3, %arg15] [%c64, %c64] [%c512, %c1_1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
          %7 = air.dma_memcpy_nd async [%async_token_11, %arg16] (%results_12[] [] [], %arg13[%arg15, %results_5] [%c64, %c64] [%c512, %c1_1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
          %8 = air.herd @herd_0 async [%7, %6]  tile (%arg17, %arg18) in (%arg19=%c4, %arg20=%c4) args(%arg21=%results_10, %arg22=%results_7) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 3 : i32} {
            %c1_15 = arith.constant 1 : index
            %c0_16 = arith.constant 0 : index
            %c64_17 = arith.constant 64 : index
            %c16 = arith.constant 16 : index
            %c32 = arith.constant 32 : index
            %async_token_18, %results_19 = air.execute -> (index) {
              %13 = affine.apply #map1()[%arg17]
              air.execute_terminator %13 : index
            }
            %async_token_20, %results_21 = air.execute -> (index) {
              %13 = affine.apply #map1()[%arg18]
              air.execute_terminator %13 : index
            }
            %9 = air.wait_all async [%async_token_20, %async_token_18] 
            %async_token_22, %results_23 = air.execute -> (memref<16x16xbf16, 2>) {
              %alloc = memref.alloc() : memref<16x16xbf16, 2>
              air.execute_terminator %alloc : memref<16x16xbf16, 2>
            }
            %10 = air.dma_memcpy_nd async [%async_token_22, %9] (%results_23[] [] [], %arg22[%results_19, %results_21] [%c16, %c16] [%c64_17, %c1_15]) {id = 4 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
            %async_token_25, %results_26 = air.execute [%10] -> (memref<16x16xbf16, 2>) {
              %alloc = memref.alloc() : memref<16x16xbf16, 2>
              air.execute_terminator %alloc : memref<16x16xbf16, 2>
            }
            %async_token_27, %results_28 = air.execute [%10] -> (memref<16x16xbf16, 2>) {
              %alloc = memref.alloc() : memref<16x16xbf16, 2>
              air.execute_terminator %alloc : memref<16x16xbf16, 2>
            }
            %wait_all_1 = air.wait_all async [%async_token_25, %async_token_27]
            %async_token_32, %results_33 = air.execute [%10] -> (memref<16x16xbf16, 2>) {
              %alloc = memref.alloc() : memref<16x16xbf16, 2>
              air.execute_terminator %alloc : memref<16x16xbf16, 2>
            }
            %async_token_34, %results_35 = air.execute [%10] -> (memref<16x16xbf16, 2>) {
              %alloc = memref.alloc() : memref<16x16xbf16, 2>
              air.execute_terminator %alloc : memref<16x16xbf16, 2>
            }
            %wait_all_2 = air.wait_all async [%async_token_32, %async_token_34]
            %11:3 = scf.for %arg23 = %c0_16 to %c64_17 step %c32 iter_args(%arg24 = %wait_all_1, %arg25 = %wait_all_2, %arg26 = %wait_all_2) -> (!air.async.token, !air.async.token, !air.async.token) {
              %13 = air.dma_memcpy_nd async [%arg24] (%results_26[] [] [], %arg21[%results_19, %arg23] [%c16, %c16] [%c64_17, %c1_15]) {id = 5 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
              %async_token_29 = air.execute [%13, %arg26] {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_26, %results_28 : memref<16x16xbf16, 2>, memref<16x16xbf16, 2>) outs(%results_23 : memref<16x16xbf16, 2>)
              }
              %async_token_36, %results_37 = air.execute [%13] -> (index) {
                %15 = affine.apply #map2()[%arg23]
                air.execute_terminator %15 : index
              }
              %14 = air.dma_memcpy_nd async [%arg25, %async_token_36] (%results_33[] [] [], %arg21[%results_19, %results_37] [%c16, %c16] [%c64_17, %c1_15]) {id = 7 : i32} : (memref<16x16xbf16, 2>, memref<64x64xbf16, 1>)
              %async_token_38 = air.execute [%14, %async_token_29] {              
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_33, %results_35 : memref<16x16xbf16, 2>, memref<16x16xbf16, 2>) outs(%results_23 : memref<16x16xbf16, 2>)
              }
              scf.yield %async_token_29, %async_token_38, %async_token_38 : !air.async.token, !air.async.token, !air.async.token
            }
            %async_token_30 = air.execute [%11#2] {
              memref.dealloc %results_26 : memref<16x16xbf16, 2>
            }
            %async_token_31 = air.execute [%11#2] {
              memref.dealloc %results_28 : memref<16x16xbf16, 2>
            }
            %async_token_39 = air.execute [%11#2] {
              memref.dealloc %results_33 : memref<16x16xbf16, 2>
            }
            %async_token_40 = air.execute [%11#2] {
              memref.dealloc %results_35 : memref<16x16xbf16, 2>
            }
            %12 = air.dma_memcpy_nd async [%11#0, %11#1, %11#2] (%arg22[%results_19, %results_21] [%c16, %c16] [%c64_17, %c1_15], %results_23[] [] []) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<16x16xbf16, 2>)
            %async_token_24 = air.execute [%12] {
              memref.dealloc %results_23 : memref<16x16xbf16, 2>
            }
          }
          %async_token_13 = air.execute [%8] {
            memref.dealloc %results_10 : memref<64x64xbf16, 1>
          }
          %async_token_14 = air.execute [%8] {
            memref.dealloc %results_12 : memref<64x64xbf16, 1>
          }
          scf.yield %8 : !air.async.token
        }
        %5 = air.dma_memcpy_nd async [%4] (%arg14[%results_3, %results_5] [%c64, %c64] [%c512, %c1_1], %results_7[] [] []) {id = 10 : i32} : (memref<512x512xbf16>, memref<64x64xbf16, 1>)
        %async_token_8 = air.execute [%5] {
          memref.dealloc %results_7 : memref<64x64xbf16, 1>
        }
      }
    }
    return
  }
}

