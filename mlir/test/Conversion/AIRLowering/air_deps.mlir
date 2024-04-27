//===- air_deps.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-std %s | FileCheck %s

// CHECK-LABEL: func.func @execute
// CHECK: %[[V0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
// CHECK: %[[E0:.*]] = airrt.wait_all : !airrt.event
// CHECK: airrt.wait_all %[[E0]]
// CHECK: memref.dealloc %[[V0]] : memref<64x64xi32>
// CHECK: %[[E1:.*]] = airrt.wait_all : !airrt.event
// CHECK: airrt.wait_all %[[E1]]
func.func @execute() {
  %0, %1 = air.execute -> (memref<64x64xi32>) {
    %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    air.execute_terminator %1 : memref<64x64xi32>
  }
  %2 = air.execute [%0] {
    memref.dealloc %1: memref<64x64xi32>
  }
  air.wait_all [%2]
  return
}

// CHECK-LABEL: func.func @scf_for
// CHECK: %[[V0:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[V1:.*]] = scf.for %arg0 = %c0 to %c64 step %c1 iter_args(%[[V3:.*]] = %[[V0]]) -> (!airrt.event) {
// CHECK:   %[[V2:.*]] = airrt.wait_all %[[V3]] : !airrt.event
// CHECK:   scf.yield %[[V2]] : !airrt.event
// CHECK: airrt.wait_all %[[V1]]
func.func @scf_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %0 = air.wait_all async
  %1 = scf.for %arg10 = %c0 to %c64 step %c1 iter_args(%iter_arg = %0) -> (!air.async.token) {
    %2 = air.wait_all async [%iter_arg]
    scf.yield %2 : !air.async.token
  }
  air.wait_all [%1]
  return
}

// CHECK-LABEL: func.func @scf_par_execute
// CHECK: %[[V0:.*]] = airrt.wait_all : !airrt.event
// CHECK: affine.for %[[V1:.*]] = 0 to 64 {
// CHECK:   %[[V2:.*]] = arith.muli %arg0, %c4 : index
// CHECK:   %[[V3:.*]] = airrt.wait_all : !airrt.event
// CHECK: }
// CHECK: airrt.wait_all
func.func @scf_par_execute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %4 = air.wait_all async
  %3 = scf.parallel (%arg8) = (%c0) to (%c64) step (%c1) init (%4) -> !air.async.token {
    %0, %1 = air.execute -> (index) {
      %c4_3 = arith.constant 4 : index
      %8 = arith.muli %arg8, %c4_3 : index
      air.execute_terminator %8 : index
    }
    scf.reduce(%0 : !air.async.token) {
    ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
      %9 = air.wait_all async [%arg10, %arg11] 
      scf.reduce.return %9 : !air.async.token
    }
  }
  air.wait_all [%3]
  return
}

// CHECK-LABEL: func.func @scf_if
// CHECK: %[[V0:.*]] = scf.if {{.*}} -> (!airrt.event) {
// CHECK:   %[[V1:.*]] = airrt.wait_all : !airrt.event
// CHECK:   scf.yield %[[V1]] : !airrt.event
// CHECK: } else {
// CHECK:   %[[V2:.*]] = airrt.wait_all : !airrt.event
// CHECK:   scf.yield %[[V2]] : !airrt.event
// CHECK: airrt.wait_all %[[V0]]
func.func @scf_if(%0 : i1) {
  %1 = scf.if %0 -> (!air.async.token) {
    %2 = air.wait_all async
    scf.yield %2 : !air.async.token
  } else {
    %2 = air.wait_all async
    scf.yield %2 : !air.async.token
  }
  air.wait_all [%1]
  return
}

// CHECK-LABEL: func.func @scf_for_par
// CHECK: %[[V0:.*]] = scf.for {{.*}} iter_args(%[[V1:.*]] = %{{.*}}) -> (!airrt.event) {
// CHECK:   %[[V2:.*]] = scf.parallel {{.*}} init (%[[V1]]) -> !airrt.event {
// CHECK:     %[[V3:.*]] = airrt.wait_all : !airrt.event
// CHECK:     %[[V4:.*]] = airrt.wait_all %[[V1]], %[[V3]] : !airrt.event
// CHECK:     scf.reduce(%[[V4]] : !airrt.event) {
// CHECK:   scf.yield %[[V2]] : !airrt.event
#map5 = affine_map<()[s0] -> (s0 * 32)>
func.func @scf_for_par(%arg0: memref<1024x512xi8>, %arg1: memref<512x1024xi8>, %arg2: memref<1024x1024xi32>, %arg3: memref<1024x1024xi32>) {
  %c16 = arith.constant 16 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c16, %arg7=%c16) args(%arg8=%arg3, %arg9=%arg0, %arg10=%arg1, %arg11=%arg2) : memref<1024x1024xi32>, memref<1024x512xi8>, memref<512x1024xi8>, memref<1024x1024xi32> attributes {id = 1 : i32} {
    %6 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
      %c16_22 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c32_23 = arith.constant 32 : index
      %c64_24 = arith.constant 64 : index
      %c0_25 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_26 = arith.constant 1 : index
      %8 = air.wait_all async 
      %14 = scf.for %arg12 = %c0_25 to %c16_22 step %c1_26 iter_args(%arg13 = %8) -> (!air.async.token) {
        %29 = scf.parallel (%arg14, %arg15) = (%c0_25, %c0_25) to (%c2, %c2) step (%c1_26, %c1_26) init (%arg13) -> !air.async.token {
          %async_token_57, %results_58 = air.execute -> (index) {
            %31 = affine.apply #map5()[%arg14]
            air.execute_terminator %31 : index
          }
          %30 = air.wait_all async [%arg13, %async_token_57] 
          scf.reduce(%30 : !air.async.token) {
          ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
            %31 = air.wait_all async [%arg16, %arg17] 
            scf.reduce.return %31 : !air.async.token
          }
        }
        scf.yield %29 : !air.async.token
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}





// #map8 = affine_map<()[s0] -> (s0 * 32)>
//   func.func @scf_pars(%arg0: memref<1024x512xi8>, %arg1: memref<512x1024xi8>, %arg2: memref<1024x1024xi32>, %arg3: memref<1024x1024xi32>) {
//     %c16 = arith.constant 16 : index
//     %0 = air.launch async (%arg4, %arg5) in (%arg6=%c16, %arg7=%c16) args(%arg8=%arg3, %arg9=%arg0, %arg10=%arg1, %arg11=%arg2) : memref<1024x1024xi32>, memref<1024x512xi8>, memref<512x1024xi8>, memref<1024x1024xi32> attributes {id = 1 : i32} {
//       %6 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
//         %c16_22 = arith.constant 16 : index
//         %c256 = arith.constant 256 : index
//         %c8 = arith.constant 8 : index
//         %c128 = arith.constant 128 : index
//         %c4 = arith.constant 4 : index
//         %c32_23 = arith.constant 32 : index
//         %c64_24 = arith.constant 64 : index
//         %c0_25 = arith.constant 0 : index
//         %c2 = arith.constant 2 : index
//         %c1_26 = arith.constant 1 : index
//         %7 = air.wait_all async 
//         %8 = air.wait_all async 
//         %10 = scf.parallel (%arg12, %arg13) = (%c0_25, %c0_25) to (%c2, %c2) step (%c1_26, %c1_26) init (%7) -> !air.async.token {
//           %async_token_57, %results_58 = air.execute -> (index) {
//             %30 = affine.apply #map8()[%arg13]
//             air.execute_terminator %30 : index
//           }
//           %29 = air.wait_all async [%7, %async_token_57] 
//           scf.reduce(%29 : !air.async.token) {
//           ^bb0(%arg14: !air.async.token, %arg15: !air.async.token):
//             %30 = air.wait_all async [%arg14, %arg15] 
//             scf.reduce.return %30 : !air.async.token
//           }
//         }
//         %14 = scf.for %arg12 = %c0_25 to %c16_22 step %c1_26 iter_args(%arg13 = %10) -> (!air.async.token) {
//           %29 = scf.parallel (%arg14, %arg15) = (%c0_25, %c0_25) to (%c2, %c2) step (%c1_26, %c1_26) init (%arg13) -> !air.async.token {
//             %async_token_57, %results_58 = air.execute -> (index) {
//               %31 = affine.apply #map8()[%arg14]
//               air.execute_terminator %31 : index
//             }
//             %30 = air.wait_all async [%arg13, %async_token_57] 
//             scf.reduce(%30 : !air.async.token) {
//             ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
//               %31 = air.wait_all async [%arg16, %arg17] 
//               scf.reduce.return %31 : !air.async.token
//             }
//           }
//           scf.yield %29 : !air.async.token
//         }
//         air.segment_terminator
//       }
//       air.launch_terminator
//     }
//     return
//   }

