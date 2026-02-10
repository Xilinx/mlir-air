//===- air_channel_to_locks_core_to_core.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// one-to-one communication
// CHECK: aie.device
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 1)
// CHECK:         %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0)
// CHECK:         %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 1)
// CHECK:         %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 0)
// CHECK:         %[[VAL_7:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_8:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>

// CHECK:    aie.mem(%[[VAL_2]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_7]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_2]])  {
// CHECK:           aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.mem(%[[VAL_1]])  {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[VAL_8]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }

// CHECK:    aie.core(%[[VAL_1]])  {
// CHECK:           aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:           aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_0 [1, 1]
func.func @one_to_one() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// two-to-two parallel dataflow
// CHECK: aie.device
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:         %[[VAL_3:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_4:.*]] = aie.tile(3, 4)
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_4]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = aie.buffer(%[[VAL_3]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_15:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>

// CHECK:         aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)

#set1 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
air.channel @channel_1 [1, 2]
func.func @two_to_two() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_1[%c0, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_1[%c0, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// one-to-two core-to-core broadcast
// CHECK: aie.device
// CHECK:         %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:         %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:         %[[VAL_3:.*]] = aie.tile(2, 4)
// CHECK:         %[[VAL_4:.*]] = aie.tile(3, 4)
// CHECK:         %[[VAL_13:.*]] = aie.buffer(%[[VAL_4]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_14:.*]] = aie.buffer(%[[VAL_3]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_15:.*]] = aie.buffer(%[[VAL_2]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[VAL_16:.*]] = aie.buffer(%[[VAL_1]]) {{{.*}}} : memref<32x32xbf16, 2>

// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_2]], DMA : 1)
// CHECK:         aie.flow(%[[VAL_3]], DMA : 0, %[[VAL_4]], DMA : 1)

#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 == 0, s1 == 1)>
air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
air.channel @channel_3 [1, 1] {broadcast_shape = [1, 2]}
func.func @one_to_two() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_2[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %5 = affine.if #set3()[%arg8, %arg9] -> !air.async.token {
            %6 = air.channel.put async [%async_token_6]  @channel_3[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %6 : !air.async.token
          } else {
            %6 = air.channel.get async [%async_token_6]  @channel_2[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            %7 = air.channel.get async [%6]  @channel_3[%arg8, %arg9] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
            affine.yield %7 : !air.async.token
          }
          affine.yield %5 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// Core-to-core cascade flow
// CHECK: aie.device
// CHECK:         %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         %[[tile_2_5:.*]] = aie.tile(2, 5)
// CHECK:         %[[tile_2_6:.*]] = aie.tile(2, 6)
// CHECK:         aie.core(%[[tile_2_6]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           linalg.add
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[%[[arg6]]] [16] [1]
// CHECK:             %[[vecread:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[vecread]]
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_5]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[%[[arg6]]] [16] [1]
// CHECK:             %[[cascade_get:.*]] = aie.get_cascade()
// CHECK-NEXT:        vector.transfer_write %[[cascade_get]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[%[[arg6]]] [16] [1]
// CHECK:             %[[vecread:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[vecread]]
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[%[[arg6]]] [16] [1]
// CHECK:             %[[cascade_get:.*]] = aie.get_cascade()
// CHECK-NEXT:        vector.transfer_write %[[cascade_get]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[%[[arg6]]] [16] [1]
// CHECK:             %[[vecread:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[vecread]]
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_3]])
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[%[[arg6]]] [16] [1]
// CHECK:             %[[cascade_get:.*]] = aie.get_cascade()
// CHECK-NEXT:        vector.transfer_write %[[cascade_get]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add

// CHECK:         aie.cascade_flow(%[[tile_2_6]], %[[tile_2_5]])
// CHECK:         aie.cascade_flow(%[[tile_2_5]], %[[tile_2_4]])
// CHECK:         aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

#set = affine_set<()[s0] : (s0 - 3 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
air.channel @channel_0 [3] {channel_type = "cascade"}
air.channel @channel_1 [1]
air.channel @channel_2 [1]
func.func @cascade(%arg0: memref<2048xi32>, %arg1: memref<2048xi32>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<2048xi32>, memref<2048xi32> attributes {id = 1 : i32} {
    %c4 = arith.constant 4 : index
    %c1_0 = arith.constant 1 : index
    %1 = air.channel.put async  @channel_1[] (%arg6[] [] []) {id = 1 : i32} : (memref<2048xi32>)
    %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c4) attributes {id = 2 : i32} {
      %c1_1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %async_token, %results = air.execute -> (memref<2048xi32, 2 : i32>) {
        %alloc = memref.alloc() : memref<2048xi32, 2 : i32>
        air.execute_terminator %alloc : memref<2048xi32, 2 : i32>
      }
      %async_token_2 = air.execute [%async_token] {
        linalg.fill ins(%c1_i32 : i32) outs(%results : memref<2048xi32, 2 : i32>)
      }
      %4 = affine.if #set()[%arg9] -> !air.async.token {
        %async_token_3, %results_4 = air.execute -> (memref<2048xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<2048xi32, 2 : i32>
          air.execute_terminator %alloc : memref<2048xi32, 2 : i32>
        }
        %5 = air.channel.get async [%async_token_3]  @channel_1[] (%results_4[] [] []) {id = 2 : i32} : (memref<2048xi32, 2 : i32>)
        %async_token_5 = air.execute [%5, %async_token_2] {
          linalg.add ins(%results_4, %results : memref<2048xi32, 2 : i32>, memref<2048xi32, 2 : i32>) outs(%results : memref<2048xi32, 2 : i32>)
        }
        %6 = arith.subi %arg9, %c1_1 : index
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c2048 = arith.constant 2048 : index
        %7 = scf.for %arg600 = %c0 to %c2048 step %c16 iter_args (%iterarg = %async_token_5) -> !air.async.token {
          %subview_12 = memref.subview %results[%arg600] [16] [1] : memref<2048xi32, 2 : i32> to memref<16xi32, strided<[1], offset: ?>, 2 : i32>
          %700 = air.channel.put async [%iterarg]  @channel_0[%6] (%subview_12[] [] []) {id = 3 : i32} : (memref<16xi32, strided<[1], offset: ?>, 2 : i32>)
          scf.yield %700 : !air.async.token
        }
        affine.yield %7 : !air.async.token
      } else {
        %5 = affine.if #set1()[%arg9] -> !air.async.token {
          %async_token_3, %results_4 = air.execute -> (memref<2048xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<2048xi32, 2 : i32>
            air.execute_terminator %alloc : memref<2048xi32, 2 : i32>
          }
          %6 = arith.subi %arg9, %c1_1 : index
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c2048 = arith.constant 2048 : index
          %7 = scf.for %arg600 = %c0 to %c2048 step %c16 iter_args (%iterarg = %async_token_3) -> !air.async.token {
            %subview_12 = memref.subview %results_4[%arg600] [16] [1] : memref<2048xi32, 2 : i32> to memref<16xi32, strided<[1], offset: ?>, 2 : i32>
            %700 = air.channel.get async [%iterarg]  @channel_0[%arg9] (%subview_12[] [] []) {id = 3 : i32} : (memref<16xi32, strided<[1], offset: ?>, 2 : i32>)
            scf.yield %700 : !air.async.token
          }
          %async_token_5 = air.execute [%7, %async_token_2] {
            linalg.add ins(%results_4, %results : memref<2048xi32, 2 : i32>, memref<2048xi32, 2 : i32>) outs(%results : memref<2048xi32, 2 : i32>)
          }
          %8 = scf.for %arg600 = %c0 to %c2048 step %c16 iter_args (%iterarg = %async_token_5) -> !air.async.token {
            %subview_12 = memref.subview %results[%arg600] [16] [1] : memref<2048xi32, 2 : i32> to memref<16xi32, strided<[1], offset: ?>, 2 : i32>
            %700 = air.channel.put async [%iterarg]  @channel_0[%6] (%subview_12[] [] []) {id = 3 : i32} : (memref<16xi32, strided<[1], offset: ?>, 2 : i32>)
            scf.yield %700 : !air.async.token
          }
          affine.yield %8 : !air.async.token
        } else {
          %async_token_3, %results_4 = air.execute -> (memref<2048xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<2048xi32, 2 : i32>
            air.execute_terminator %alloc : memref<2048xi32, 2 : i32>
          }
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c2048 = arith.constant 2048 : index
          %6 = scf.for %arg600 = %c0 to %c2048 step %c16 iter_args (%iterarg = %async_token_3) -> !air.async.token {
            %subview_12 = memref.subview %results_4[%arg600] [16] [1] : memref<2048xi32, 2 : i32> to memref<16xi32, strided<[1], offset: ?>, 2 : i32>
            %700 = air.channel.get async [%iterarg]  @channel_0[%arg9] (%subview_12[] [] []) {id = 3 : i32} : (memref<16xi32, strided<[1], offset: ?>, 2 : i32>)
            scf.yield %700 : !air.async.token
          }
          %async_token_5 = air.execute [%6, %async_token_2] {
            linalg.add ins(%results_4, %results : memref<2048xi32, 2 : i32>, memref<2048xi32, 2 : i32>) outs(%results : memref<2048xi32, 2 : i32>)
          }
          %7 = air.channel.put async [%async_token_5]  @channel_2[] (%results[] [] []) {id = 7 : i32} : (memref<2048xi32, 2 : i32>)
          affine.yield %7 : !air.async.token
        }
        affine.yield %async_token_2 : !air.async.token
      }
    }
    %3 = air.channel.get async  @channel_2[] (%arg7[] [] []) {id = 8 : i32} : (memref<2048xi32>)
  }
  return
}

// -----

// Core-to-core cascade flow; collapse memref shape using memref.collapse_shape, to enforce 1D vector for aie.put/get_cascade.
// CHECK: aie.device
// CHECK:         %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         %[[tile_2_5:.*]] = aie.tile(2, 5)
// CHECK:         %[[tile_2_6:.*]] = aie.tile(2, 6)
// CHECK:         aie.core(%[[tile_2_6]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           linalg.add
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[0, 0, %[[arg6]]] [1, 1, 16] [1, 1, 1]
// CHECK:             %[[collapse_shape:.*]] = memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK-NEXT:        %[[vecread:.*]] = vector.transfer_read %[[collapse_shape]][%c0], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[vecread]]
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_5]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[0, 0, %[[arg6]]] [1, 1, 16] [1, 1, 1]
// CHECK:             %[[cascade_get:.*]] = aie.get_cascade()
// CHECK-NEXT:        %[[collapse_shape:.*]] = memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK-NEXT:        vector.transfer_write %[[cascade_get]], %[[collapse_shape]][%c0] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[0, 0, %[[arg6]]] [1, 1, 16] [1, 1, 1]
// CHECK:             %[[collapse_shape:.*]] = memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK-NEXT:        %[[vecread:.*]] = vector.transfer_read %[[collapse_shape]][%c0], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[vecread]]
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[0, 0, %[[arg6]]] [1, 1, 16] [1, 1, 1]
// CHECK:             %[[cascade_get:.*]] = aie.get_cascade()
// CHECK-NEXT:        %[[collapse_shape:.*]] = memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK-NEXT:        vector.transfer_write %[[cascade_get]], %[[collapse_shape]][%c0] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[0, 0, %[[arg6]]] [1, 1, 16] [1, 1, 1]
// CHECK:             %[[collapse_shape:.*]] = memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK-NEXT:        %[[vecread:.*]] = vector.transfer_read %[[collapse_shape]][%c0], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[vecread]]
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_3]])
// CHECK:           scf.for %[[arg6:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview_12:.*]] = memref.subview %{{.*}}[0, 0, %[[arg6]]] [1, 1, 16] [1, 1, 1]
// CHECK:             %[[cascade_get:.*]] = aie.get_cascade()
// CHECK-NEXT:        %[[collapse_shape:.*]] = memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK-NEXT:        vector.transfer_write %[[cascade_get]], %[[collapse_shape]][%c0] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add

// CHECK:         aie.cascade_flow(%[[tile_2_6]], %[[tile_2_5]])
// CHECK:         aie.cascade_flow(%[[tile_2_5]], %[[tile_2_4]])
// CHECK:         aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

#set = affine_set<()[s0] : (s0 - 3 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
module {
  air.channel @channel_0 [3] {channel_type = "cascade"}
  air.channel @channel_1 [1]
  air.channel @channel_2 [1]
  func.func @cascade2(%arg0: memref<1x1x2048xi32>, %arg1: memref<1x1x2048xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<1x1x2048xi32>, memref<1x1x2048xi32> attributes {id = 1 : i32} {
      %c4 = arith.constant 4 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.channel.put async  @channel_1[] (%arg6[] [] []) {id = 1 : i32} : (memref<1x1x2048xi32>)
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c4) attributes {id = 2 : i32} {
        %c2048 = arith.constant 2048 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c1_i32 = arith.constant 1 : i32
        %async_token, %results = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
        }
        %async_token_2 = air.execute [%async_token] {
          linalg.fill ins(%c1_i32 : i32) outs(%results : memref<1x1x2048xi32, 2 : i32>)
        }
        %4 = affine.if #set()[%arg9] -> !air.async.token {
          %async_token_3, %results_4 = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
            air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
          }
          %5 = air.channel.get async [%async_token_3]  @channel_1[] (%results_4[] [] []) {id = 2 : i32} : (memref<1x1x2048xi32, 2 : i32>)
          %async_token_5 = air.execute [%5, %async_token_2] {
            linalg.add ins(%results_4, %results : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%results : memref<1x1x2048xi32, 2 : i32>)
          }
          %6 = arith.subi %arg9, %c1_1 : index
          %7 = scf.for %arg12 = %c0 to %c2048 step %c16 iter_args(%arg13 = %async_token_5) -> (!air.async.token) {
            %subview = memref.subview %results[0, 0, %arg12] [1, 1, 16] [1, 1, 1] : memref<1x1x2048xi32, 2 : i32> to memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>
            %8 = air.channel.put async [%arg13]  @channel_0[%6] (%subview[] [] []) {id = 3 : i32} : (memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>)
            scf.yield %8 : !air.async.token
          }
          affine.yield %7 : !air.async.token
        } else {
          %5 = affine.if #set1()[%arg9] -> !air.async.token {
            %async_token_3, %results_4 = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
              air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
            }
            %6 = arith.subi %arg9, %c1_1 : index
            %7 = scf.for %arg12 = %c0 to %c2048 step %c16 iter_args(%arg13 = %async_token_3) -> (!air.async.token) {
              %subview = memref.subview %results_4[0, 0, %arg12] [1, 1, 16] [1, 1, 1] : memref<1x1x2048xi32, 2 : i32> to memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>
              %10 = air.channel.get async [%arg13]  @channel_0[%arg9] (%subview[] [] []) {id = 4 : i32} : (memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>)
              scf.yield %10 : !air.async.token
            }
            %async_token_5 = air.execute [%async_token_3, %async_token_2] {
              linalg.add ins(%results_4, %results : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%results : memref<1x1x2048xi32, 2 : i32>)
            }
            %8 = scf.for %arg12 = %c0 to %c2048 step %c16 iter_args(%arg13 = %async_token_5) -> (!air.async.token) {
              %subview = memref.subview %results[0, 0, %arg12] [1, 1, 16] [1, 1, 1] : memref<1x1x2048xi32, 2 : i32> to memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>
              %10 = air.channel.put async [%arg13]  @channel_0[%6] (%subview[] [] []) {id = 5 : i32} : (memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>)
              scf.yield %10 : !air.async.token
            }
            %9 = air.wait_all async [%7, %8] 
            affine.yield %9 : !air.async.token
          } else {
            %async_token_3, %results_4 = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
              air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
            }
            %6 = scf.for %arg12 = %c0 to %c2048 step %c16 iter_args(%arg13 = %async_token_3) -> (!air.async.token) {
              %subview = memref.subview %results_4[0, 0, %arg12] [1, 1, 16] [1, 1, 1] : memref<1x1x2048xi32, 2 : i32> to memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>
              %9 = air.channel.get async [%arg13]  @channel_0[%arg9] (%subview[] [] []) {id = 6 : i32} : (memref<1x1x16xi32, strided<[2048, 2048, 1], offset: ?>, 2 : i32>)
              scf.yield %9 : !air.async.token
            }
            %async_token_5 = air.execute [%async_token_3, %async_token_2] {
              linalg.add ins(%results_4, %results : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%results : memref<1x1x2048xi32, 2 : i32>)
            }
            %7 = air.channel.put async [%async_token_5]  @channel_2[] (%results[] [] []) {id = 7 : i32} : (memref<1x1x2048xi32, 2 : i32>)
            %8 = air.wait_all async [%6, %7] 
            affine.yield %8 : !air.async.token
          }
          affine.yield %async_token_2 : !air.async.token
        }
      }
      %3 = air.channel.get async  @channel_2[] (%arg7[] [] []) {id = 8 : i32} : (memref<1x1x2048xi32>)
    }
    return
  }
}

// -----

// Test cascade flattening with 2D memref (32x64 = 2048 elements, same total as 1D test)
// The memref is flattened to 1D before tiling for cascade transfer
// CHECK: aie.device
// CHECK:         %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1]
// CHECK:           scf.for %[[arg:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK:             %[[subview:.*]] = memref.subview %{{.*}}[%[[arg]]] [16] [1]
// CHECK:             aie.put_cascade
// CHECK:           aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

// Test 2D memref flattening for cascade
#set_2d = affine_set<()[s0] : (s0 - 1 == 0)>
module {
  air.channel @cascade_2d [1] {channel_type = "cascade"}
  func.func @cascade_2d_flatten(%arg0: memref<32x64xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x64xi32> attributes {id = 1 : i32} {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {id = 2 : i32} {
        %c1_i32 = arith.constant 1 : i32
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<32x64xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<32x64xi32, 2 : i32>
          air.execute_terminator %alloc : memref<32x64xi32, 2 : i32>
        }
        %async_token_2 = air.execute [%async_token] {
          linalg.fill ins(%c1_i32 : i32) outs(%results : memref<32x64xi32, 2 : i32>)
        }
        %2 = affine.if #set_2d()[%arg9] -> !air.async.token {
          %3 = arith.subi %arg9, %c1_1 : index
          %4 = air.channel.put async [%async_token_2]  @cascade_2d[%3] (%results[] [] []) {id = 3 : i32} : (memref<32x64xi32, 2 : i32>)
          affine.yield %4 : !air.async.token
        } else {
          %3 = air.channel.get async [%async_token_2]  @cascade_2d[%arg9] (%results[] [] []) {id = 4 : i32} : (memref<32x64xi32, 2 : i32>)
          affine.yield %3 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Test cascade flattening with 4D memref (2x4x8x32 = 2048 elements)
// The memref is flattened from 4D to 1D before tiling for cascade transfer
// CHECK: aie.device
// CHECK:         %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2, 3]
// CHECK:           scf.for %[[arg:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg]]] [16] [1]
// CHECK:             aie.put_cascade
// CHECK:           aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

// Test 4D memref flattening for cascade
#set_4d = affine_set<()[s0] : (s0 - 1 == 0)>
module {
  air.channel @cascade_4d [1] {channel_type = "cascade"}
  func.func @cascade_4d_flatten(%arg0: memref<2x4x8x32xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<2x4x8x32xi32> attributes {id = 1 : i32} {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {id = 2 : i32} {
        %c1_i32 = arith.constant 1 : i32
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<2x4x8x32xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<2x4x8x32xi32, 2 : i32>
          air.execute_terminator %alloc : memref<2x4x8x32xi32, 2 : i32>
        }
        %async_token_2 = air.execute [%async_token] {
          linalg.fill ins(%c1_i32 : i32) outs(%results : memref<2x4x8x32xi32, 2 : i32>)
        }
        %2 = affine.if #set_4d()[%arg9] -> !air.async.token {
          %3 = arith.subi %arg9, %c1_1 : index
          %4 = air.channel.put async [%async_token_2]  @cascade_4d[%3] (%results[] [] []) {id = 3 : i32} : (memref<2x4x8x32xi32, 2 : i32>)
          affine.yield %4 : !air.async.token
        } else {
          %3 = air.channel.get async [%async_token_2]  @cascade_4d[%arg9] (%results[] [] []) {id = 4 : i32} : (memref<2x4x8x32xi32, 2 : i32>)
          affine.yield %3 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Test cascade with bf16 element type (cascade width 512 bits = 32 bf16 elements per tile)
// CHECK: aie.device
// CHECK:         %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1]
// CHECK:           scf.for %[[arg:.*]] = %c0{{.*}} to %c1024{{.*}} step %c32{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg]]] [32] [1]
// CHECK:             aie.put_cascade
// CHECK:           aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

// Test bf16 cascade flattening (different tile size due to element width)
#set_bf16 = affine_set<()[s0] : (s0 - 1 == 0)>
module {
  air.channel @cascade_bf16 [1] {channel_type = "cascade"}
  func.func @cascade_bf16_flatten(%arg0: memref<32x32xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x32xbf16> attributes {id = 1 : i32} {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {id = 2 : i32} {
        %cst = arith.constant 1.0 : bf16
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_2 = air.execute [%async_token] {
          linalg.fill ins(%cst : bf16) outs(%results : memref<32x32xbf16, 2>)
        }
        %2 = affine.if #set_bf16()[%arg9] -> !air.async.token {
          %3 = arith.subi %arg9, %c1_1 : index
          %4 = air.channel.put async [%async_token_2]  @cascade_bf16[%3] (%results[] [] []) {id = 3 : i32} : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %3 = air.channel.get async [%async_token_2]  @cascade_bf16[%arg9] (%results[] [] []) {id = 4 : i32} : (memref<32x32xbf16, 2>)
          affine.yield %3 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// Core-to-core cascade flow; vectorizing channel.put/get with for loops, to fulfill the AIE cascade width requirment.
// With pre-flattening: the memref is collapsed first, then tiled with a single 1D scf.for loop.
// CHECK: aie.device
// CHECK:         %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:         %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         %[[tile_2_5:.*]] = aie.tile(2, 5)
// CHECK:         %[[tile_2_6:.*]] = aie.tile(2, 6)
// CHECK:         aie.core(%[[tile_2_6]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           linalg.add
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK:           scf.for %[[arg0:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg0]]] [16] [1]
// CHECK:             %[[cascade_data:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[cascade_data]] : vector<16xi32>)
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_5]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK:           scf.for %[[arg0:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg0]]] [16] [1]
// CHECK:             %[[cascade_data:.*]] = aie.get_cascade() : vector<16xi32>
// CHECK-NEXT:        vector.transfer_write %[[cascade_data]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK:           scf.for %[[arg0:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg0]]] [16] [1]
// CHECK:             %[[cascade_data:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[cascade_data]] : vector<16xi32>)
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           %[[CST:.*]] = arith.constant 0 : i32
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK:           scf.for %[[arg0:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg0]]] [16] [1]
// CHECK:             %[[cascade_data:.*]] = aie.get_cascade() : vector<16xi32>
// CHECK-NEXT:        vector.transfer_write %[[cascade_data]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK:           scf.for %[[arg0:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg0]]] [16] [1]
// CHECK:             %[[cascade_data:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %[[CST]] {in_bounds = [true]}
// CHECK-NEXT:        aie.put_cascade(%[[cascade_data]] : vector<16xi32>)
// CHECK-NEXT:      }
// CHECK:         aie.core(%[[tile_2_3]])
// CHECK:           memref.collapse_shape %{{.*}} {{.*}}[0, 1, 2]
// CHECK:           scf.for %[[arg0:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK-NEXT:        %[[subview:.*]] = memref.subview %{{.*}}[%[[arg0]]] [16] [1]
// CHECK:             %[[cascade_data:.*]] = aie.get_cascade() : vector<16xi32>
// CHECK-NEXT:        vector.transfer_write %[[cascade_data]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}
// CHECK-NEXT:      }
// CHECK:           linalg.add

// CHECK:         aie.cascade_flow(%[[tile_2_6]], %[[tile_2_5]])
// CHECK:         aie.cascade_flow(%[[tile_2_5]], %[[tile_2_4]])
// CHECK:         aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

// Test pre-flattening with full memref (not pre-tiled in IR)
#set = affine_set<()[s0] : (s0 - 3 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
module {
  air.channel @channel_0 [3] {channel_type = "cascade"}
  air.channel @channel_1 [1]
  air.channel @channel_2 [1]
  func.func @cascade3(%arg0: memref<1x1x2048xi32>, %arg1: memref<1x1x2048xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<1x1x2048xi32>, memref<1x1x2048xi32> attributes {id = 1 : i32} {
      %c4 = arith.constant 4 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.channel.put async  @channel_1[] (%arg6[] [] []) {id = 1 : i32} : (memref<1x1x2048xi32>)
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c4) attributes {id = 2 : i32} {
        %c1_1 = arith.constant 1 : index
        %c1_i32 = arith.constant 1 : i32
        %async_token, %results = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
          air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
        }
        %async_token_2 = air.execute [%async_token] {
          linalg.fill ins(%c1_i32 : i32) outs(%results : memref<1x1x2048xi32, 2 : i32>)
        }
        %4 = affine.if #set()[%arg9] -> !air.async.token {
          %async_token_3, %results_4 = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
            air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
          }
          %5 = air.channel.get async [%async_token_3]  @channel_1[] (%results_4[] [] []) {id = 2 : i32} : (memref<1x1x2048xi32, 2 : i32>)
          %async_token_5 = air.execute [%5, %async_token_2] {
            linalg.add ins(%results_4, %results : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%results : memref<1x1x2048xi32, 2 : i32>)
          }
          %6 = arith.subi %arg9, %c1_1 : index
          %7 = air.channel.put async [%async_token_5]  @channel_0[%6] (%results[] [] []) {id = 3 : i32} : (memref<1x1x2048xi32, 2 : i32>)
          affine.yield %7 : !air.async.token
        } else {
          %5 = affine.if #set1()[%arg9] -> !air.async.token {
            %async_token_3, %results_4 = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
              air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
            }
            %6 = arith.subi %arg9, %c1_1 : index
            %7 = air.channel.get async [%async_token_3]  @channel_0[%arg9] (%results_4[] [] []) {id = 4 : i32} : (memref<1x1x2048xi32, 2 : i32>)
            %async_token_5 = air.execute [%7, %async_token_2] {
              linalg.add ins(%results_4, %results : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%results : memref<1x1x2048xi32, 2 : i32>)
            }
            %8 = air.channel.put async [%async_token_5]  @channel_0[%6] (%results[] [] []) {id = 5 : i32} : (memref<1x1x2048xi32, 2 : i32>)
            affine.yield %8 : !air.async.token
          } else {
            %async_token_3, %results_4 = air.execute -> (memref<1x1x2048xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
              air.execute_terminator %alloc : memref<1x1x2048xi32, 2 : i32>
            }
            %6 = air.channel.get async [%async_token_3]  @channel_0[%arg9] (%results_4[] [] []) {id = 6 : i32} : (memref<1x1x2048xi32, 2 : i32>)
            %async_token_5 = air.execute [%6, %async_token_2] {
              linalg.add ins(%results_4, %results : memref<1x1x2048xi32, 2 : i32>, memref<1x1x2048xi32, 2 : i32>) outs(%results : memref<1x1x2048xi32, 2 : i32>)
            }
            %7 = air.channel.put async [%async_token_5]  @channel_2[] (%results[] [] []) {id = 7 : i32} : (memref<1x1x2048xi32, 2 : i32>)
            affine.yield %7 : !air.async.token
          }
          affine.yield %async_token_2 : !air.async.token
        }
      }
      %3 = air.channel.get async  @channel_2[] (%arg7[] [] []) {id = 8 : i32} : (memref<1x1x2048xi32>)
    }
    return
  }
}
