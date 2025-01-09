//===- fuse_alloc_dealloc.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-fuse-alloc-dealloc %s | FileCheck %s

// Fuse allocs and deallocs into the inner-most region that contains all uses of the memref.

module {

// Fuse alloc/dealloc into an scf.for.

// CHECK-LABEL:   @func0
// CHECK:   scf.for{{.*}}{
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.channel.put
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }
// CHECK:   }

  func.func @func0() {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %async_token_0, %results_1 = air.execute -> (memref<4x1x64x64xbf16>) {
      %alloc = memref.alloc() : memref<4x1x64x64xbf16>
      air.execute_terminator %alloc : memref<4x1x64x64xbf16>
    }
    %5 = scf.for %arg7 = %c0 to %c512 step %c256 iter_args(%arg8 = %async_token_0) -> (!air.async.token) {
      %6 = air.channel.put async [%arg8]  @channel_0[] (%results_1[] [] []) : (memref<4x1x64x64xbf16>)
      scf.yield %6 : !air.async.token
    }
    %async_token_2 = air.execute [%5] {
      memref.dealloc %results_1 : memref<4x1x64x64xbf16>
    }
    return
  }

// Fuse alloc/dealloc into an scf.for loop nest.

// CHECK-LABEL:   @func1
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.channel.put
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }
// CHECK:   }
// CHECK:   }

  func.func @func1() {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %async_token_0, %results_1 = air.execute -> (memref<4x1x64x64xbf16>) {
      %alloc = memref.alloc() : memref<4x1x64x64xbf16>
      air.execute_terminator %alloc : memref<4x1x64x64xbf16>
    }
    %4 = scf.for %arg5 = %c0 to %c512 step %c256 iter_args(%arg6 = %async_token_0) -> (!air.async.token) {
      %5 = scf.for %arg7 = %c0 to %c512 step %c256 iter_args(%arg8 = %arg6) -> (!air.async.token) {
        %6 = air.channel.put async [%arg8]  @channel_0[] (%results_1[] [] []) : (memref<4x1x64x64xbf16>)
        scf.yield %6 : !air.async.token
      }
      scf.yield %5 : !air.async.token
    }
    %async_token_2 = air.execute [%4] {
      memref.dealloc %results_1 : memref<4x1x64x64xbf16>
    }
    return
  }

// Two loop nests.

// CHECK-LABEL:   @func2
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.channel.put
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }
// CHECK:   }
// CHECK:   }
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.channel.put
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }
// CHECK:   }
// CHECK:   }

  func.func @func2() {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %async_token_0, %results_1 = air.execute -> (memref<4x1x64x64xbf16>) {
      %alloc = memref.alloc() : memref<4x1x64x64xbf16>
      air.execute_terminator %alloc : memref<4x1x64x64xbf16>
    }
    %async_token_2, %results_3 = air.execute -> (memref<4x1x64x64xbf16>) {
      %alloc = memref.alloc() : memref<4x1x64x64xbf16>
      air.execute_terminator %alloc : memref<4x1x64x64xbf16>
    }
    %4 = scf.for %arg5 = %c0 to %c512 step %c256 iter_args(%arg6 = %async_token_0) -> (!air.async.token) {
      %5 = scf.for %arg7 = %c0 to %c512 step %c256 iter_args(%arg8 = %arg6) -> (!air.async.token) {
        %6 = air.channel.put async [%arg8]  @channel_0[] (%results_1[] [] []) : (memref<4x1x64x64xbf16>)
        scf.yield %6 : !air.async.token
      }
      scf.yield %5 : !air.async.token
    }
    %7 = scf.for %arg5 = %c0 to %c512 step %c256 iter_args(%arg6 = %async_token_2) -> (!air.async.token) {
      %5 = scf.for %arg7 = %c0 to %c512 step %c256 iter_args(%arg8 = %arg6) -> (!air.async.token) {
        %6 = air.channel.put async [%arg8]  @channel_1[] (%results_3[] [] []) : (memref<4x1x64x64xbf16>)
        scf.yield %6 : !air.async.token
      }
      scf.yield %5 : !air.async.token
    }
    %async_token_4 = air.execute [%4] {
      memref.dealloc %results_1 : memref<4x1x64x64xbf16>
    }
    %async_token_5 = air.execute [%7] {
      memref.dealloc %results_3 : memref<4x1x64x64xbf16>
    }
    return
  }

// Two loop nests (non-async version).

// CHECK-LABEL:   @func3
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   memref.alloc()
// CHECK:   air.channel.put
// CHECK:   memref.dealloc
// CHECK:   }
// CHECK:   }
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   memref.alloc()
// CHECK:   air.channel.put
// CHECK:   memref.dealloc
// CHECK:   }
// CHECK:   }

  func.func @func3() {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %results_1 = memref.alloc() : memref<4x1x64x64xbf16>
    %results_3 = memref.alloc() : memref<4x1x64x64xbf16>
    scf.for %arg5 = %c0 to %c512 step %c256 {
      scf.for %arg7 = %c0 to %c512 step %c256 {
        air.channel.put @channel_0[] (%results_1[] [] []) : (memref<4x1x64x64xbf16>)
      }
    }
    scf.for %arg5 = %c0 to %c512 step %c256 {
      scf.for %arg7 = %c0 to %c512 step %c256 {
        air.channel.put @channel_1[] (%results_3[] [] []) : (memref<4x1x64x64xbf16>)
      }
    }
    memref.dealloc %results_1 : memref<4x1x64x64xbf16>
    memref.dealloc %results_3 : memref<4x1x64x64xbf16>
    return
  }

// Fuse alloc/dealloc into air.herd.

// CHECK-LABEL:   @func4
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   linalg.fill
// CHECK-NEXT:   }
// CHECK:   scf.for{{.*}}{
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   linalg.fill
// CHECK-NEXT:   }
// CHECK:   }
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   linalg.fill
// CHECK-NEXT:   }
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }
// CHECK:   }
// CHECK:   }

  func.func @func4() {

    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %async_token, %results = air.execute -> (memref<4x4x16x16x4x4xbf16, 2 : i32>) {
      %alloc = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<4x4x16x16x4x4xbf16, 2 : i32>
    }
    %3 = air.herd @herd_0 async [%async_token]  tile (%arg5, %arg6) in (%arg7=%c4, %arg8=%c4) args(%arg9=%results) : memref<4x4x16x16x4x4xbf16, 2 : i32> attributes {id = 4 : i32, link_with = "mm.o"} {
      %c1_4 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c256_5 = arith.constant 256 : index
      %c0_6 = arith.constant 0 : index
      %c512_7 = arith.constant 512 : index
      %5 = air.wait_all async 
      %6 = scf.for %arg10 = %c0_6 to %c512_7 step %c256_5 iter_args(%arg11 = %5) -> (!air.async.token) {
        %7 = scf.for %arg12 = %c0_6 to %c512_7 step %c256_5 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %subview = memref.subview %arg9[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
          %async_token_8 = air.execute [%arg13] {
            linalg.fill ins(%cst : bf16) outs(%subview : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
          }
          %9 = scf.for %arg14 = %c0_6 to %c8 step %c1_4 iter_args(%arg15 = %async_token_8) -> (!air.async.token) {
            %subview_11 = memref.subview %arg9[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
            %async_token_12 = air.execute [%arg15] {
              linalg.fill ins(%cst : bf16) outs(%subview_11 : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
            }
            scf.yield %async_token_12 : !air.async.token
          }
          %subview_9 = memref.subview %arg9[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
          %async_token_10 = air.execute [%9] {
            linalg.fill ins(%cst : bf16) outs(%subview_9 : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
          }
          %8 = air.wait_all async [%async_token_10]
          scf.yield %8 : !air.async.token
        }
        scf.yield %7 : !air.async.token
      }
    }
    %async_token_2 = air.execute [%3] {
      memref.dealloc %results : memref<4x4x16x16x4x4xbf16, 2 : i32>
    }
    return
  }

// Fuse into air.herd (non-async version).

// CHECK-LABEL:   @func5
// CHECK:   scf.for{{.*}}{
// CHECK:   scf.for{{.*}}{
// CHECK:   memref.alloc()
// CHECK:   linalg.fill
// CHECK:   scf.for{{.*}}{
// CHECK:   linalg.fill
// CHECK:   }
// CHECK:   linalg.fill
// CHECK:   memref.dealloc
// CHECK:   }
// CHECK:   }

  func.func @func5() {

    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %results = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
    air.herd @herd_0 tile (%arg5, %arg6) in (%arg7=%c4, %arg8=%c4) args(%arg9=%results) : memref<4x4x16x16x4x4xbf16, 2 : i32> attributes {id = 4 : i32, link_with = "mm.o"} {
      %c1_4 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c256_5 = arith.constant 256 : index
      %c0_6 = arith.constant 0 : index
      %c512_7 = arith.constant 512 : index
      %5 = air.wait_all async 
      scf.for %arg10 = %c0_6 to %c512_7 step %c256_5 {
        scf.for %arg12 = %c0_6 to %c512_7 step %c256_5 {
          %subview = memref.subview %arg9[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
          linalg.fill ins(%cst : bf16) outs(%subview : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
          scf.for %arg14 = %c0_6 to %c8 step %c1_4 {
            %subview_11 = memref.subview %arg9[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
            linalg.fill ins(%cst : bf16) outs(%subview_11 : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
          }
          %subview_9 = memref.subview %arg9[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
          linalg.fill ins(%cst : bf16) outs(%subview_9 : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
        }
      }
    }
    memref.dealloc %results : memref<4x4x16x16x4x4xbf16, 2 : i32>
    return
  }

// Check if dealloc's dependencies get preserved.

// CHECK-LABEL:   @func6
// CHECK:   %[[VALUE0:.*]], %{{.*}} = air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   linalg.fill
// CHECK-NEXT:   }
// CHECK:   air.execute [%[[VALUE0]]] {
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }

  func.func @func6() {

    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %wait_all = air.wait_all async
    %async_token, %results = air.execute [%wait_all] -> (memref<4x4x16x16x4x4xbf16, 2 : i32>) {
      %alloc = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<4x4x16x16x4x4xbf16, 2 : i32>
    }
    %3 = air.herd @herd_0 async [%async_token]  tile (%arg5, %arg6) in (%arg7=%c4, %arg8=%c4) args(%arg9=%results) : memref<4x4x16x16x4x4xbf16, 2 : i32> attributes {id = 4 : i32, link_with = "mm.o"} {
      %c1_4 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c256_5 = arith.constant 256 : index
      %c0_6 = arith.constant 0 : index
      %c512_7 = arith.constant 512 : index
      %async_token_8 = air.execute {
        linalg.fill ins(%cst : bf16) outs(%arg9 : memref<4x4x16x16x4x4xbf16, 2 : i32>)
      }
    }
    %async_token_2 = air.execute [%3, %async_token] {
      memref.dealloc %results : memref<4x4x16x16x4x4xbf16, 2 : i32>
    }
    return
  }

// Scf.for nested inside an air.herd.
  
// CHECK-LABEL:   @func7
// CHECK:   scf.for{{.*}}{
// CHECK:   %[[VALUE0:.*]], %{{.*}} = air.execute{{.*}}{
// CHECK-NEXT:   memref.alloc()
// CHECK-NEXT:   air.execute_terminator
// CHECK-NEXT:   }
// CHECK:   air.execute [%[[VALUE0]], %{{.*}}] {
// CHECK-NEXT:   linalg.fill
// CHECK-NEXT:   }
// CHECK:   air.execute{{.*}}{
// CHECK-NEXT:   memref.dealloc
// CHECK-NEXT:   }
// CHECK:   }

  func.func @func7() {

    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %wait_all = air.wait_all async
    %async_token, %results = air.execute [%wait_all] -> (memref<4x4x16x16x4x4xbf16, 2 : i32>) {
      %alloc = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<4x4x16x16x4x4xbf16, 2 : i32>
    }
    %3 = air.herd @herd_0 async [%async_token]  tile (%arg5, %arg6) in (%arg7=%c4, %arg8=%c4) args(%arg9=%results) : memref<4x4x16x16x4x4xbf16, 2 : i32> attributes {id = 4 : i32, link_with = "mm.o"} {
      %c1_4 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c256_5 = arith.constant 256 : index
      %c0_6 = arith.constant 0 : index
      %c512_7 = arith.constant 512 : index
      %5 = air.wait_all async 
      %6 = scf.for %arg10 = %c0_6 to %c512_7 step %c256_5 iter_args(%arg11 = %5) -> (!air.async.token) {
        %async_token_8 = air.execute [%arg11] {
          linalg.fill ins(%cst : bf16) outs(%arg9 : memref<4x4x16x16x4x4xbf16, 2 : i32>)
        }
        scf.yield %async_token_8 : !air.async.token
      }
    }
    %async_token_2 = air.execute [%3, %async_token] {
      memref.dealloc %results : memref<4x4x16x16x4x4xbf16, 2 : i32>
    }
    return
  }
}
