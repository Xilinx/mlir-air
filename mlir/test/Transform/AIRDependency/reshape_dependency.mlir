//===-reshape_dependency.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

//
// Test that the dependence token are correcly generated
// for a series of alloc, reshape/expand/collapse and 
// compute and dealloc ops.
//
module {
  func.func @forward(%arg0: memref<512x512xf32>) -> memref<512x512xf32> {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %alloc_0 = memref.alloc() : memref<2xindex>
    memref.store %c64, %alloc_0[%c0] : memref<2xindex>
    memref.store %c64, %alloc_0[%c1] : memref<2xindex>
    air.launch (%arg1, %arg2) in (%arg3=%c4, %arg4=%c4) args(%arg5=%alloc_0) : memref<2xindex> {
      air.segment @forward_0  args(%arg6=%arg5) : memref<2xindex> {
        %c2 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%arg6) : memref<2xindex> {
          %c0_1 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c32 = arith.constant 32 : index
          %alloc_1 = memref.alloc() : memref<64x64xf32, 2>
          scf.for %arg25 = %c0_1 to %c32 step %c16 {
            %alloc_2 = memref.alloc() : memref<64x64xbf16, 2>
            // CHECK: %[[ASYNC_TOKEN_0:.*]], %[[VAL_0:.*]] = air.execute -> (memref<64x64xbf16, 2>) {
            // CHECK-NEXT:%[[ALLOC0:.*]] =  memref.alloc() : memref<64x64xbf16, 2>
            // CHECK-NEXT: air.execute_terminator %[[ALLOC0]]
             %alloc_3 = memref.alloc() : memref<16x4x64xbf16, 2>
            // CHECK: %[[ASYNC_TOKEN_1:.*]], %[[VAL_1:.*]] = air.execute -> (memref<16x4x64xbf16, 2>) {
            // CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc() : memref<16x4x64xbf16, 2>
            // CHECK-NEXT: air.execute_terminator %[[ALLOC1]]
            %reshape = memref.reshape %alloc_3(%arg11) : (memref<16x4x64xbf16, 2>, memref<2xindex>) -> memref<64x64xbf16, 2>
            linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_2, %reshape : memref<64x64xbf16, 2>, memref<64x64xbf16, 2>) outs(%alloc_1 : memref<64x64xf32, 2>)
            // CHECK: %[[ASYNC_TOKEN_2:.*]] = air.execute [%[[ASYNC_TOKEN_1]], %[[ASYNC_TOKEN_0]]
            // CHECK-NEXT: memref.reshape
            // CHECK-NEXT: linalg.matmul
             memref.dealloc %alloc_2 : memref<64x64xbf16, 2>
            // CHECK: %[[ASYNC_TOKEN_3:.*]] = air.execute [%[[ASYNC_TOKEN_2]]]
            // CHECK-NEXT: memref.dealloc %[[VAL_0]]
            memref.dealloc %alloc_3 : memref<16x4x64xbf16, 2>
            // CHECK: %[[ASYNC_TOKEN_4:.*]] = air.execute [%[[ASYNC_TOKEN_2]]]
            // CHECK-NEXT: memref.dealloc %[[VAL_1]]
            // CHECK: air.wait_all async [{{.*}}, %[[ASYNC_TOKEN_2]]]
          }
          memref.dealloc %alloc_1 : memref<64x64xf32, 2>
        }
      }
    }
    return %arg0 : memref<512x512xf32>
  }
}
