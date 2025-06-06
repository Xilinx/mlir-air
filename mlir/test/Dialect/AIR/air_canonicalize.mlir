//===- air_herd_launch_canonicalize.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize  %s | FileCheck %s

// CHECK-LABEL: func.func @herd
// CHECK: air.herd tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) {
func.func @herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  return
}

// CHECK-LABEL: func.func @herd_async
// CHECK: air.herd async [{{.*}}] tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) attributes {attr_name = "attrValue"} {
func.func @herd_async(%arg0: i32, %e0 : !air.async.token) {
  %cst2 = arith.constant 2 : index
  %e1 = air.herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { attr_name="attrValue" } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: herd_async_1
// CHECK: air.execute
// CHECK: memref.alloc
// CHECK: air.execute_terminator
// CHECK: air.herd async [{{.*}}] tile (
// CHECK: air.dma_memcpy_nd
func.func @herd_async_1() {
  %cst2 = arith.constant 2 : index
  %t0, %results = air.execute -> (memref<1xi32>) {
    %1 = memref.alloc() : memref<1xi32>
    air.execute_terminator %1 : memref<1xi32>
  }
  %t1 = air.wait_all async [%t0]
  %e0 = air.wait_all async [%t0, %t1]
  %e1 = air.herd async [%t0, %t1, %e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%results) : memref<1xi32> {
    %d0 = air.dma_memcpy_nd async (%op0[] [] [], %op0[] [] []) : (memref<1xi32>, memref<1xi32>)
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: herd_async_2
// CHECK: air.herd async  tile (
func.func @herd_async_2() {
  %cst2 = arith.constant 2 : index
  %t0 = air.wait_all async
  %t1 = air.wait_all async [%t0]
  %e0 = air.wait_all async [%t0, %t1]
  %e1 = air.herd async [%t0, %t1, %e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: segment_async
// CHECK: air.segment @segment_0 async
// CHECK: %[[TOKEN0:.*]] = air.wait_all async 
// CHECK-NEXT: %[[TOKEN1:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN4:.*]] = %[[TOKEN0]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN2:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN5:.*]] = %[[TOKEN4]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN3:.*]] = air.herd @herd_0 async [%[[TOKEN5]]]  tile () in () 
// CHECK-NEXT: scf.yield %[[TOKEN3]] : !air.async.token
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[TOKEN2]] : !air.async.token
// CHECK-NEXT: }
func.func @segment_async() {
  %7 = air.segment @segment_0 async  {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %async_token, %results = air.execute -> (memref<4x4x64x64xbf16, 1 : i32>) {
      %alloc = memref.alloc() : memref<4x4x64x64xbf16, 1 : i32>
      air.execute_terminator %alloc : memref<4x4x64x64xbf16, 1 : i32>
    }
    %8 = air.wait_all async
    %9 = scf.for %arg8 = %c0 to %c512 step %c256 iter_args(%arg9 = %8) -> (!air.async.token) {
      %10 = scf.for %arg10 = %c0 to %c512 step %c256 iter_args(%arg11 = %arg9) -> (!air.async.token) {
        %12 = air.wait_all async [%arg11, %async_token]
        %11 = air.herd @herd_0 async [%12]  tile () in () {
          %alloc = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
          memref.dealloc %alloc : memref<4x4x16x16x4x4xbf16, 2 : i32>
        }
        scf.yield %11 : !air.async.token
      }
      scf.yield %10 : !air.async.token
    }
  }
  return
}

// CHECK-LABEL: segment_async_1
// CHECK: air.segment @segment_0 async
// CHECK: %[[TOKEN0:.*]], %[[RES0:.*]] = air.execute -> (memref<4x4x64x64xbf16, 1 : i32>) {
// CHECK: %[[TOKEN2:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN5:.*]] = %[[TOKEN0]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN3:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN6:.*]] = %[[TOKEN5]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN4:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN7:.*]] = %[[TOKEN6]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN8:.*]] = air.channel.put async [%[[TOKEN7]]]
// CHECK-NEXT: scf.yield %[[TOKEN8]] : !air.async.token
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[TOKEN4]] : !air.async.token
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[TOKEN3]] : !air.async.token
// CHECK-NEXT: }
func.func @segment_async_1() {
  %7 = air.segment @segment_0 async  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %async_token, %results = air.execute -> (memref<4x4x64x64xbf16, 1 : i32>) {
      %alloc = memref.alloc() : memref<4x4x64x64xbf16, 1 : i32>
      air.execute_terminator %alloc : memref<4x4x64x64xbf16, 1 : i32>
    }
    %8 = air.wait_all async
    %29 = scf.for %arg8 = %c0 to %c512 step %c256 iter_args(%arg9 = %8) -> (!air.async.token) {
      %61 = air.wait_all async [%async_token] 
      %62 = scf.for %arg10 = %c0 to %c512 step %c256 iter_args(%arg11 = %61) -> (!air.async.token) {
        %64 = air.wait_all async [%arg9] 
        %65 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %64) -> (!air.async.token) {
          %67 = air.channel.put async [%arg11]  @channel_0[] (%results[] [] []) : (memref<4x4x64x64xbf16, 1 : i32>)
          %68 = air.wait_all async [%67] 
          scf.yield %68 : !air.async.token
        }
        %66 = air.wait_all async [%65] 
        scf.yield %66 : !air.async.token
      }
      %63 = air.wait_all async [%62] 
      scf.yield %63 : !air.async.token
    }
  }
  return
}

// CHECK-LABEL: segment_async_2
// CHECK: air.segment @segment_0 async
// CHECK: %[[TOKEN0:.*]], %[[RES0:.*]] = air.execute -> (memref<4x4x64x64xbf16, 1 : i32>) {
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT: %[[TOKEN3:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN6:.*]] = %[[TOKEN0]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN4:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TOKEN7:.*]] = %[[TOKEN6]]) -> (!air.async.token) {
// CHECK-NEXT: %[[TOKEN8:.*]] = air.channel.put async [%[[TOKEN7]]]
// CHECK-NEXT: scf.yield %[[TOKEN8]] : !air.async.token
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[TOKEN4]] : !air.async.token
// CHECK-NEXT: }
// CHECK-NEXT: }
func.func @segment_async_2() {
  %7 = air.segment @segment_0 async  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %async_token, %results = air.execute -> (memref<4x4x64x64xbf16, 1 : i32>) {
      %alloc = memref.alloc() : memref<4x4x64x64xbf16, 1 : i32>
      air.execute_terminator %alloc : memref<4x4x64x64xbf16, 1 : i32>
    }
    %8 = air.wait_all async
    scf.for %arg8 = %c0 to %c512 step %c256 {
      %61 = air.wait_all async [%async_token] 
      %62 = scf.for %arg10 = %c0 to %c512 step %c256 iter_args(%arg11 = %61) -> (!air.async.token) {
        %64 = air.wait_all async [%8] 
        %65 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %64) -> (!air.async.token) {
          %67 = air.channel.put async [%arg11]  @channel_0[] (%results[] [] []) : (memref<4x4x64x64xbf16, 1 : i32>)
          %68 = air.wait_all async [%67] 
          scf.yield %68 : !air.async.token
        }
        %66 = air.wait_all async [%65] 
        scf.yield %66 : !air.async.token
      }
      %63 = air.wait_all async [%62] 
    }
  }
  return
}

// CHECK-LABEL: wait_all_0
// CHECK-NEXT: return
func.func @wait_all_0() {
  %0 = air.wait_all async
  %1 = air.wait_all async [%0]
  air.wait_all [%0, %1]
  air.wait_all
  return
}

// CHECK-LABEL: wait_all_1
// CHECK-NEXT: return
func.func @wait_all_1() {
  %async_token, %results = air.execute -> (memref<1xi32>) {
    %1 = memref.alloc() : memref<1xi32>
    air.execute_terminator %1 : memref<1xi32>
  }
  %0 = air.wait_all async [%async_token]
  air.wait_all [%0]
  return
}

// CHECK: func.func @wait_all_2
// CHECK: scf.for
// CHECK: scf.for{{.*}}iter_args(%[[TOK0:.*]] = %{{.*}})
// CHECK: %[[GET0:.*]] = air.channel.get async [%[[TOK0]]]  @channel_0
// CHECK: %[[GET1:.*]] = air.channel.get async [%[[TOK0]]]  @channel_0
// CHECK: %[[GET2:.*]] = air.channel.get async [%[[TOK0]]]  @channel_0
// CHECK: %[[GET3:.*]] = air.channel.get async [%[[TOK0]]]  @channel_0
// CHECK: %[[PUT0:.*]] = air.channel.put async [%[[GET0]]]  @channel_1
// CHECK: %[[PUT1:.*]] = air.channel.put async [%[[GET1]]]  @channel_1
// CHECK: %[[YIELD:.*]] = air.wait_all async [%[[PUT0]], %[[PUT1]]]
// CHECK: scf.yield %[[YIELD]]
// CHECK: scf.yield

func.func @wait_all_2(%arg0: memref<1xi8>, %arg1: memref<1xi8>, %arg2: memref<1xi8>, %arg3: memref<1xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c4096 = arith.constant 4096 : index
  %c16384 = arith.constant 16384 : index
  %0 = air.wait_all async 
  %6 = scf.for %arg4 = %c0 to %c512 step %c256 iter_args(%arg5 = %0) -> (!air.async.token) {
    %7 = air.wait_all async [%arg5] 
    %8 = scf.for %arg6 = %c0 to %c512 step %c256 iter_args(%arg7 = %7) -> (!air.async.token) {
      %9 = air.channel.get async [%arg7]  @channel_0[%c0, %c0] (%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c64, %c64] [%c16384, %c4096, %c64, %c1]) {id = 6 : i32} : (memref<1xi8>)
      %10 = air.wait_all async [%9] 
      %11 = air.channel.get async [%arg7]  @channel_0[%c1, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c1, %c1, %c64, %c64] [%c16384, %c4096, %c64, %c1]) {id = 7 : i32} : (memref<1xi8>)
      %12 = air.wait_all async [%11] 
      %13 = air.channel.get async [%arg7]  @channel_0[%c2, %c0] (%arg2[%c0, %c0, %c0, %c0] [%c1, %c1, %c64, %c64] [%c16384, %c4096, %c64, %c1]) {id = 8 : i32} : (memref<1xi8>)
      %14 = air.wait_all async [%13] 
      %15 = air.channel.get async [%arg7]  @channel_0[%c3, %c0] (%arg3[%c0, %c0, %c0, %c0] [%c1, %c1, %c64, %c64] [%c16384, %c4096, %c64, %c1]) {id = 9 : i32} : (memref<1xi8>)
      %16 = air.wait_all async [%15] 
      %41 = air.wait_all async [%10, %12, %14, %16] 
      %42 = air.channel.put async [%41]  @channel_1[%c0, %c0] (%arg0[%c0, %c0, %c0, %c0] [%c1, %c64, %c4, %c64] [%c16384, %c64, %c4096, %c1]) {id = 22 : i32} : (memref<1xi8>)
      %43 = air.channel.put async [%41]  @channel_1[%c1, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c1, %c64, %c4, %c64] [%c16384, %c64, %c4096, %c1]) {id = 23 : i32} : (memref<1xi8>)
      %46 = air.wait_all async [%42, %43] 
      scf.yield %46 : !air.async.token
    }
    scf.yield %8 : !air.async.token
  }
  return
}

// CHECK-LABEL: execute_0
// CHECK-NEXT: return
func.func @execute_0() {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (index) {
    air.execute_terminator %c0 : index
  }
  return
}

// CHECK-LABEL: execute_1
// CHECK-NEXT: %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: return %{{.*}} : index
func.func @execute_1(%i : index) -> index {
  %async_token, %results:2 = air.execute -> (index, index) {
    %c0 = arith.constant 0 : index
    air.execute_terminator %i, %c0 : index, index
  }
  return %results#1 : index
}

// CHECK-LABEL: execute_2
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: air.wait_all async 
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, !air.async.token
func.func @execute_2() -> (index, !air.async.token) {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (index) {
    air.execute_terminator %c0 : index
  }
  %t = air.wait_all async [%async_token]
  return %results, %t : index, !air.async.token
}

// CHECK-LABEL: execute_3
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: air.wait_all async 
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, !air.async.token
func.func @execute_3() -> (index, !air.async.token) {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (index) {
    %1 = memref.alloc() : memref<1xi32>
    air.dma_memcpy_nd (%1[] [] [], %1[] [] []) : (memref<1xi32>, memref<1xi32>)
    air.execute_terminator %c0 : index
  }
  %t = air.wait_all async [%async_token]
  return %results, %t : index, !air.async.token
}

// CHECK-LABEL: execute_4
// CHECK: air.execute
// CHECK: memref.alloc
// CHECK: air.dma_memcpy_nd
// CHECK: air.execute_terminator
// CHECK: air.execute
// CHECK: memref.dealloc
func.func @execute_4() -> (memref<1xi32>, !air.async.token) {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (memref<1xi32>) {
    %1 = memref.alloc() : memref<1xi32>
    air.dma_memcpy_nd (%1[] [] [], %1[] [] []) : (memref<1xi32>, memref<1xi32>)
    air.execute_terminator %1 : memref<1xi32>
  }
  %t = air.wait_all async [%async_token]
  %async_token_0 = air.execute {
    memref.dealloc %results : memref<1xi32>
  }
  return %results, %t : memref<1xi32>, !air.async.token
}

// CHECK-LABEL: execute_5
// CHECK: scf.for {{.*}} {
// CHECK: air.execute {{.*}} {
// CHECK-NEXT: memref.store
// CHECK-NEXT: }
// CHECK: scf.yield
// CHECK-NEXT: }
func.func @execute_5(%alloc : memref<4xi32>) -> (!air.async.token) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 2 : i32
  %t = air.wait_all async 
  %0 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %t) -> (!air.async.token) {
    %async_token_0, %results_1 = air.execute [%arg1] -> (index) {
      air.execute_terminator %arg0 : index
    }
    %async_token_1 = air.execute [%async_token_0] {
      memref.store %cst, %alloc[%results_1] : memref<4xi32>
    }
    scf.yield %async_token_1 : !air.async.token
  }
  return %0 :!air.async.token
}

// CHECK: func.func @chan_0
// CHECK: %[[TOKEN0:.*]] = air.channel.get async  @channel_0
// CHECK: %[[TOKEN1:.*]] = air.channel.get async  @channel_1
// CHECK: air.channel.put async [%[[TOKEN0]]]
func.func @chan_0(%arg0 : memref<4x1x64x64xbf16>, %arg1 : memref<1x4x64x64xbf16>) {
  %0 = air.channel.get async  @channel_0[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  %1 = air.channel.get async  @channel_1[] (%arg1[] [] []) : (memref<1x4x64x64xbf16>)
  %2 = air.channel.put async [%0, %1]  @channel_0[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  return
}

// CHECK: func.func @chan_1
// CHECK: %[[TOKEN0:.*]] = air.channel.put async  @channel_0
// CHECK: %[[TOKEN1:.*]] = air.channel.put async  @channel_1
func.func @chan_1(%arg0 : memref<4x1x64x64xbf16>) {
  %1 = air.channel.put async  @channel_0[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  %2 = air.channel.put async [%1]  @channel_1[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  return
}

// CHECK: func.func @chan_2
// CHECK: %[[TOKEN0:.*]] = air.channel.get async  @channel_0
// CHECK: %[[TOKEN1:.*]] = air.channel.get async [%[[TOKEN0]]] @channel_1
func.func @chan_2(%arg0 : memref<4x1x64x64xbf16>) {
  %0 = air.channel.get async  @channel_0[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  %1 = air.channel.get async [%0]  @channel_1[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  return
}

// CHECK: func.func @chan_3
// CHECK: %[[TOKEN0:.*]] = air.channel.get async  @channel_0
// CHECK: %[[TOKEN1:.*]] = air.channel.put async [%[[TOKEN0]]] @channel_1
func.func @chan_3(%arg0 : memref<4x1x64x64xbf16>) {
  %0 = air.channel.get async  @channel_0[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  %1 = air.wait_all async [%0]
  %2 = air.channel.put async [%1]  @channel_1[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  return
}

// CHECK: func.func @chan_4
// CHECK: %[[TOKEN0:.*]] = air.channel.get async  @channel_0
// CHECK: %[[TOKEN1:.*]] = air.channel.get async  @channel_1
// CHECK: %[[TOKEN2:.*]] = air.channel.put async [%[[TOKEN0]]] @channel_2
// CHECK: %[[TOKEN3:.*]] = air.channel.put async [%[[TOKEN1]]] @channel_3
func.func @chan_4(%arg0 : memref<4x1x64x64xbf16>, %arg1 : memref<4x1x64x64xbf16>) {
  %0 = air.channel.get async  @channel_0[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  %1 = air.channel.get async  @channel_1[] (%arg1[] [] []) : (memref<4x1x64x64xbf16>)
  %2 = air.wait_all async [%0, %1]
  %3 = air.channel.put async [%2]  @channel_2[] (%arg0[] [] []) : (memref<4x1x64x64xbf16>)
  %4 = air.channel.put async [%2]  @channel_3[] (%arg1[] [] []) : (memref<4x1x64x64xbf16>)
  return
}

// CHECK: func.func @dma_compose_subview
// CHECK: air.dma_memcpy_nd (%{{.*}}[%c0{{.*}}, %c0{{.*}}] [%c32{{.*}}, %c32{{.*}}] [%c64{{.*}}, %c1{{.*}}], %{{.*}}[%c0{{.*}}, %c0{{.*}}] [%c32{{.*}}, %c32{{.*}}] [%c64{{.*}}, %c1{{.*}}]
func.func @dma_compose_subview() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<64x64xbf16, 2>
  %subview_4 = memref.subview %0[%c0, %c0] [32, 32] [1, 1] : memref<64x64xbf16, 1> to memref<32x32xbf16, strided<[64, 1], offset: ?>, 1>
  %subview_5 = memref.subview %1[%c0, %c0] [32, 32] [1, 1] : memref<64x64xbf16, 2> to memref<32x32xbf16, strided<[64, 1], offset: ?>, 2>
  air.dma_memcpy_nd (%subview_4[] [] [], %subview_5[] [] []) : (memref<32x32xbf16, strided<[64, 1], offset: ?>, 1>, memref<32x32xbf16, strided<[64, 1], offset: ?>, 2>)
  return 
}

// CHECK: func.func @dma_compose_transpose
// CHECK: air.dma_memcpy_nd (%{{.*}}[%c0{{.*}}, %c0{{.*}}] [%c64{{.*}}, %c128{{.*}}] [%c1{{.*}}, %c64{{.*}}], %{{.*}}[%c0{{.*}}, %c0{{.*}}] [%c64{{.*}}, %c128{{.*}}] [%c1{{.*}}, %c64{{.*}}]
func.func @dma_compose_transpose() {
  %0 = memref.alloc() : memref<128x64xbf16, 1>
  %1 = memref.alloc() : memref<128x64xbf16, 2>
  %transpose_1 = memref.transpose %0 (d0, d1) -> (d1, d0) : memref<128x64xbf16, 1> to memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 1>
  %transpose_2 = memref.transpose %1 (d0, d1) -> (d1, d0) : memref<128x64xbf16, 2> to memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 2>
  air.dma_memcpy_nd (%transpose_1[] [] [], %transpose_2[] [] []) : (memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 1>, memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 2>)
  return 
}

// CHECK: func.func @dma_compose_expand_shape
// CHECK: air.dma_memcpy_nd (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c4096{{.*}}, %c64{{.*}}, %c1{{.*}}], %{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c4096{{.*}}, %c64{{.*}}, %c1{{.*}}]
func.func @dma_compose_expand_shape() {
  %0 = memref.alloc() : memref<128x64xbf16, 1>
  %1 = memref.alloc() : memref<128x64xbf16, 2>
  %expand_shape_1 = memref.expand_shape %0 [[0, 1], [2]] output_shape [2, 64, 64] : memref<128x64xbf16, 1> into memref<2x64x64xbf16, 1>
  %expand_shape_2 = memref.expand_shape %1 [[0, 1], [2]] output_shape [2, 64, 64] : memref<128x64xbf16, 2> into memref<2x64x64xbf16, 2>
  air.dma_memcpy_nd (%expand_shape_1[] [] [], %expand_shape_2[] [] []) : (memref<2x64x64xbf16, 1>, memref<2x64x64xbf16, 2>)
  return 
}

// CHECK: func.func @dma_compose_cast
// CHECK: air.dma_memcpy_nd (%{{.*}}[%c0{{.*}}, %c0{{.*}}] [%c128{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1{{.*}}], %{{.*}}[%c0{{.*}}, %c0{{.*}}] [%c128{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1{{.*}}]
func.func @dma_compose_cast() {
  %0 = memref.alloc() : memref<128x64xbf16, 1>
  %1 = memref.alloc() : memref<128x64xbf16, 2>
  %cast = memref.cast %0 : memref<128x64xbf16, 1> to memref<128x64xbf16, strided<[64, 1], offset: ?>, 1>
  %cast_1 = memref.cast %1 : memref<128x64xbf16, 2> to memref<128x64xbf16, strided<[64, 1], offset: ?>, 2>
  air.dma_memcpy_nd (%cast[] [] [], %cast_1[] [] []) : (memref<128x64xbf16, strided<[64, 1], offset: ?>, 1>, memref<128x64xbf16, strided<[64, 1], offset: ?>, 2>)
  return 
}

// CHECK: func.func @channel_compose_subview
// CHECK: air.channel.put @channel[] ({{.*}}[{{.*}}, {{.*}}] [%c32{{.*}}, %c32{{.*}}] [%c64{{.*}}, %c1{{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async @channel[] ({{.*}}[{{.*}}, {{.*}}] [%c32{{.*}}, %c32{{.*}}] [%c64{{.*}}, %c1{{.*}}]
air.channel @channel[2,2]
func.func @channel_compose_subview() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<64x64xbf16, 2>
  %subview_4 = memref.subview %0[%c0, %c0] [32, 32] [1, 1] : memref<64x64xbf16, 1> to memref<32x32xbf16, strided<[64, 1], offset: ?>, 1>
  air.channel.put @channel[] (%subview_4[] [] []) : (memref<32x32xbf16, strided<[64, 1], offset: ?>, 1>)
  %subview_5 = memref.subview %1[%c0, %c0] [32, 32] [1, 1] : memref<64x64xbf16, 2> to memref<32x32xbf16, strided<[64, 1], offset: ?>, 2>
  %5 = air.channel.get async @channel[] (%subview_5[] [] []) : (memref<32x32xbf16, strided<[64, 1], offset: ?>, 2>)
  return 
}

// CHECK: func.func @channel_compose_transpose
// CHECK: air.channel.put @channel[] ({{.*}}[{{.*}}, {{.*}}] [%c64{{.*}}, %c128{{.*}}] [%c1{{.*}}, %c64{{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async @channel[] ({{.*}}[{{.*}}, {{.*}}] [%c64{{.*}}, %c128{{.*}}] [%c1{{.*}}, %c64{{.*}}]
func.func @channel_compose_transpose() {
  %0 = memref.alloc() : memref<128x64xbf16, 1>
  %1 = memref.alloc() : memref<128x64xbf16, 2>
  %transpose_1 = memref.transpose %0 (d0, d1) -> (d1, d0) : memref<128x64xbf16, 1> to memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 1>
  air.channel.put @channel[] (%transpose_1[] [] []) : (memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 1>)
  %transpose_2 = memref.transpose %1 (d0, d1) -> (d1, d0) : memref<128x64xbf16, 2> to memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 2>
  %5 = air.channel.get async @channel[] (%transpose_2[] [] []) : (memref<64x128xbf16, affine_map<(d0, d1) -> (d0 + d1 * 64)>, 2>)
  return 
}

// CHECK: func.func @channel_compose_expand_shape
// CHECK: air.channel.put @channel[] ({{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c4096{{.*}}, %c64{{.*}}, %c1{{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async @channel[] ({{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c4096{{.*}}, %c64{{.*}}, %c1{{.*}}]
func.func @channel_compose_expand_shape() {
  %0 = memref.alloc() : memref<128x64xbf16, 1>
  %1 = memref.alloc() : memref<128x64xbf16, 2>
  %expand_shape_1 = memref.expand_shape %0 [[0, 1], [2]] output_shape [2, 64, 64] : memref<128x64xbf16, 1> into memref<2x64x64xbf16, 1>
  air.channel.put @channel[] (%expand_shape_1[] [] []) : (memref<2x64x64xbf16, 1>)
  %expand_shape_2 = memref.expand_shape %1 [[0, 1], [2]] output_shape [2, 64, 64] : memref<128x64xbf16, 2> into memref<2x64x64xbf16, 2>
  %5 = air.channel.get async @channel[] (%expand_shape_2[] [] []) : (memref<2x64x64xbf16, 2>)
  return 
}

// CHECK: func.func @channel_compose_cast
// CHECK: air.channel.put @channel[] ({{.*}}[%c0{{.*}}, %c0{{.*}}] [%c128{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1{{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async @channel[] ({{.*}}[%c0{{.*}}, %c0{{.*}}] [%c128{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1{{.*}}]
func.func @channel_compose_cast() {
  %0 = memref.alloc() : memref<128x64xbf16, 1>
  %1 = memref.alloc() : memref<128x64xbf16, 2>
  %cast = memref.cast %0 : memref<128x64xbf16, 1> to memref<128x64xbf16, strided<[64, 1], offset: ?>, 1>
  air.channel.put @channel[] (%cast[] [] []) : (memref<128x64xbf16, strided<[64, 1], offset: ?>, 1>)
  %cast_1 = memref.cast %1 : memref<128x64xbf16, 2> to memref<128x64xbf16, strided<[64, 1], offset: ?>, 2>
  %5 = air.channel.get async @channel[] (%cast_1[] [] []) : (memref<128x64xbf16, strided<[64, 1], offset: ?>, 2>)
  return 
}

// Memref op chain on air::DmaMemcpyNdOp's src/dst memrefs.
// CHECK: func.func @func0
// CHECK:  %[[CST128:.*]] = arith.constant 128 : index
// CHECK:  %[[CST32:.*]] = arith.constant 32 : index
// CHECK:  %[[CST8:.*]] = arith.constant 8 : index
// CHECK:  %[[CST16:.*]] = arith.constant 16 : index
// CHECK:  %[[CST0:.*]] = arith.constant 0 : index
// CHECK:  %[[CST1:.*]] = arith.constant 1 : index
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %[[CST0]]] [%[[CST8]], %[[CST16]]] [%[[CST16]], %[[CST1]]]) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[CST0]], %{{.*}}] [%[[CST16]], %[[CST16]]] [%[[CST32]], %[[CST1]]]) : (memref<1x1x16x16xi32, 1>, memref<16x32xi32>)
// CHECK:  air.herd @herd_0
// CHECK:  %[[CST32_0:.*]] = arith.constant 32 : index
// CHECK:  %[[CST256_0:.*]] = arith.constant 256 : index
// CHECK:  %[[CST4_0:.*]] = arith.constant 4 : index
// CHECK:  %[[CST2_0:.*]] = arith.constant 2 : index
// CHECK:  %[[CST1_0:.*]] = arith.constant 1 : index
// CHECK:  %[[CST16_0:.*]] = arith.constant 16 : index
// CHECK:  %[[CST64_0:.*]] = arith.constant 64 : index
// CHECK:  %[[CST8_0:.*]] = arith.constant 8 : index
// CHECK:  %[[CST128_0:.*]] = arith.constant 128 : index
// CHECK:  %[[CST0_0:.*]] = arith.constant 0 : index
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]]] [%[[CST1_0]], %[[CST1_0]], %[[CST2_0]], %[[CST2_0]], %[[CST4_0]], %[[CST8_0]]] [%[[CST128_0]], %[[CST128_0]], %[[CST8_0]], %[[CST64_0]], %[[CST16_0]], %[[CST1_0]]]) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[CST0_0]], %{{.*}}, %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]]] [%[[CST1_0]], %[[CST1_0]], %[[CST2_0]], %[[CST2_0]], %[[CST8_0]], %[[CST8_0]]] [%[[CST256_0]], %[[CST256_0]], %[[CST8_0]], %[[CST128_0]], %[[CST16_0]], %[[CST1_0]]]) : (memref<1x1x2x2x8x8xi32, 2>, memref<1x1x16x16xi32, 1>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}, %[[CST0_0]], %[[CST0_0]]] [%[[CST1_0]], %[[CST1_0]], %[[CST8_0]], %[[CST16_0]]] [%[[CST128_0]], %[[CST128_0]], %[[CST16_0]], %[[CST1_0]]], %{{.*}}[%[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]], %[[CST0_0]]] [%[[CST1_0]], %[[CST1_0]], %[[CST2_0]], %[[CST4_0]], %[[CST2_0]], %[[CST8_0]]] [%[[CST128_0]], %[[CST128_0]], %[[CST32_0]], %[[CST8_0]], %[[CST64_0]], %[[CST1_0]]]) : (memref<1x1x8x16xi32, 1>, memref<1x1x2x2x4x8xi32, 2>)
// CHECK:  air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}] [%[[CST8]], %[[CST16]]] [%[[CST32]], %[[CST1]]], %{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST8]], %[[CST16]]] [%[[CST128]], %[[CST128]], %[[CST16]], %[[CST1]]]) : (memref<8x32xi32>, memref<1x1x8x16xi32, 1>)

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
func.func @func0(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<8x16xi32>, memref<16x32xi32>, memref<8x32xi32> {
    air.segment @segment_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<8x16xi32>, memref<16x32xi32>, memref<8x32xi32> {
      %c1_0 = arith.constant 1 : index
      %0 = affine.apply #map()[%arg10]
      %1 = affine.apply #map1()[%arg11]
      %subview = memref.subview %arg12[%0, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
      %subview_1 = memref.subview %arg13[0, %1] [16, 16] [1, 1] : memref<16x32xi32> to memref<16x16xi32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %arg14[%0, %1] [8, 16] [1, 1] : memref<8x32xi32> to memref<8x16xi32, strided<[32, 1], offset: ?>>
      %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
      air.dma_memcpy_nd (%alloc[] [] [], %subview[] [] []) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32, strided<[16, 1], offset: ?>>)
      %alloc_3 = memref.alloc() : memref<1x1x16x16xi32, 1>
      air.dma_memcpy_nd (%alloc_3[] [] [], %subview_1[] [] []) : (memref<1x1x16x16xi32, 1>, memref<16x16xi32, strided<[32, 1], offset: ?>>)
      %alloc_4 = memref.alloc() : memref<1x1x8x16xi32, 1>
      air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c1_0, %arg18=%c1_0) args(%arg19=%alloc, %arg20=%alloc_3, %arg21=%alloc_4) : memref<1x1x8x16xi32, 1>, memref<1x1x16x16xi32, 1>, memref<1x1x8x16xi32, 1> {
        %subview_6 = memref.subview %arg19[%arg15, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
        %subview_7 = memref.subview %arg20[0, %arg16, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xi32, 1> to memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1>
        %subview_8 = memref.subview %arg21[%arg15, %arg16, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
        %alloc_9 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
        %expand_shape = memref.expand_shape %subview_6 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 2, 4, 2, 8] : memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1> into memref<1x1x2x4x2x8xi32, strided<[128, 128, 64, 16, 8, 1], offset: ?>, 1>
        %transpose_10 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x4x2x8xi32, strided<[128, 128, 64, 16, 8, 1], offset: ?>, 1> to memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1], offset: ?>, 1>
        air.dma_memcpy_nd (%alloc_9[] [] [], %transpose_10[] [] []) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1], offset: ?>, 1>)
        %alloc_11 = memref.alloc() : memref<1x1x2x2x8x8xi32, 2>
        %expand_shape_12 = memref.expand_shape %subview_7 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 2, 8, 2, 8] : memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1> into memref<1x1x2x8x2x8xi32, strided<[256, 256, 128, 16, 8, 1], offset: ?>, 1>
        %transpose_13 = memref.transpose %expand_shape_12 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x8x2x8xi32, strided<[256, 256, 128, 16, 8, 1], offset: ?>, 1> to memref<1x1x2x2x8x8xi32, strided<[256, 256, 8, 128, 16, 1], offset: ?>, 1>
        air.dma_memcpy_nd (%alloc_11[] [] [], %transpose_13[] [] []) : (memref<1x1x2x2x8x8xi32, 2>, memref<1x1x2x2x8x8xi32, strided<[256, 256, 8, 128, 16, 1], offset: ?>, 1>)
        %alloc_14 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
        %transpose_15 = memref.transpose %alloc_14 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x2x2x4x8xi32, 2> to memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>
        air.dma_memcpy_nd (%subview_8[] [] [], %transpose_15[] [] []) : (memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>, memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>)
        memref.dealloc %alloc_9 : memref<1x1x2x2x4x8xi32, 2>
        memref.dealloc %alloc_11 : memref<1x1x2x2x8x8xi32, 2>
        memref.dealloc %alloc_14 : memref<1x1x2x2x4x8xi32, 2>
      }
      %subview_5 = memref.subview %alloc_4[0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<8x16xi32, 1>
      %transpose = memref.transpose %subview_5 (d0, d1) -> (d0, d1) : memref<8x16xi32, 1> to memref<8x16xi32, strided<[16, 1]>, 1>
      air.dma_memcpy_nd (%subview_2[] [] [], %transpose[] [] []) : (memref<8x16xi32, strided<[32, 1], offset: ?>>, memref<8x16xi32, strided<[16, 1]>, 1>)
      memref.dealloc %alloc_3 : memref<1x1x16x16xi32, 1>
      memref.dealloc %alloc : memref<1x1x8x16xi32, 1>
      memref.dealloc %alloc_4 : memref<1x1x8x16xi32, 1>
    }
  }
  return
}

// CHECK: func.func @func1
// CHECK:  air.herd @herd_0 {{.*}} args(%[[ARG0:.*]]=%{{.*}}, %[[ARG1:.*]]=%{{.*}})
// CHECK-DAG:    %[[CST4:.*]] = arith.constant 4 : index
// CHECK-DAG:    %[[CST3:.*]] = arith.constant 3 : index
// CHECK-DAG:    %[[CST1:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[CST8:.*]] = arith.constant 8 : index
// CHECK-DAG:    %[[CST64:.*]] = arith.constant 64 : index
// CHECK-DAG:    %[[CST256:.*]] = arith.constant 256 : index
// CHECK-DAG:    %[[CST768:.*]] = arith.constant 768 : index
// CHECK-DAG:    %[[CST0:.*]] = arith.constant 0 : index
// CHECK:    air.dma_memcpy_nd (%[[ARG1]][] [] [], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST3]], %[[CST3]], %[[CST4]], %[[CST1]], %[[CST8]], %[[CST8]]] [%[[CST768]], %[[CST256]], %[[CST64]], %[[CST8]], %[[CST8]], %[[CST1]]]) : (memref<3x3x4x1x8x8xi8, 2 : i32>, memref<3x3x32x8xi8, 1 : i32>)
// CHECK:  }

func.func @func1() {
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  air.launch (%arg0, %arg1, %arg2, %arg3) in (%arg4=%c2, %arg5=%c3, %arg6=%c3, %arg7=%c8) {
    air.segment @segment_0  {
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %alloc = memref.alloc() : memref<3x3x4x1x8x8xi8, 2 : i32>
      %alloc_0 = memref.alloc() : memref<3x3x32x8xi8, 1 : i32>
      air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c1) args(%arg12=%alloc_0, %arg13=%alloc) : memref<3x3x32x8xi8, 1 : i32>, memref<3x3x4x1x8x8xi8, 2 : i32> {
        %cast = memref.cast %arg12 : memref<3x3x32x8xi8, 1 : i32> to memref<3x3x32x8xi8, strided<[768, 256, 8, 1], offset: ?>, 1 : i32>
        %expand_shape = memref.expand_shape %cast [[0], [1], [2, 3], [4, 5]] output_shape [3, 3, 4, 8, 1, 8] : memref<3x3x32x8xi8, strided<[768, 256, 8, 1], offset: ?>, 1 : i32> into memref<3x3x4x8x1x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32>
        %transpose = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5) : memref<3x3x4x8x1x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32> to memref<3x3x4x1x8x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32>
        air.dma_memcpy_nd (%arg13[] [] [], %transpose[] [] []) : (memref<3x3x4x1x8x8xi8, 2 : i32>, memref<3x3x4x1x8x8xi8, strided<[768, 256, 64, 8, 8, 1], offset: ?>, 1 : i32>)
      }
    }
  }
  return
}

// CHECK: func.func @func2
// CHECK:  %[[TOK0:.*]], %[[RES0:.*]] = air.execute
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: air.execute_terminator
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2048{{.*}} step %c128{{.*}} iter_args(%[[TOK1:.*]] = %[[TOK0]])
// CHECK-NEXT: air.channel.put async [%[[TOK1]]]  @channel_3

func.func @func2(%arg0: memref<2048xi8>, %arg1: memref<2048x1024xi8>, %arg2: memref<1024xi32>) {
  %c4096 = arith.constant 4096 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c256_2 = arith.constant 256 : index
  %c1_3 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_4 = arith.constant 0 : index
  %c2048_5 = arith.constant 2048 : index
  %c128_6 = arith.constant 128 : index
  %async_token_7, %results_8 = air.execute -> (memref<2048xi8, 1>) {
    %alloc = memref.alloc() : memref<2048xi8, 1>
    air.execute_terminator %alloc : memref<2048xi8, 1>
  }
  %async_token_9, %results_10 = air.execute -> (memref<2048x256xi8, 1>) {
    %alloc = memref.alloc() : memref<2048x256xi8, 1>
    air.execute_terminator %alloc : memref<2048x256xi8, 1>
  }
  %async_token_11, %results_12 = air.execute -> (memref<256xi32, 1>) {
    %alloc = memref.alloc() : memref<256xi32, 1>
    air.execute_terminator %alloc : memref<256xi32, 1>
  }
  %async_token_13, %results_14 = air.execute -> (index) {
    air.execute_terminator %c0_4 : index
  }
  %10 = air.wait_all async [%async_token_7, %async_token_9, %async_token_11, %async_token_13] 
  %11 = scf.for %arg8 = %c0_4 to %c2048_5 step %c128_6 iter_args(%arg9 = %10) -> (!air.async.token) {
    %21 = air.channel.put async [%arg9]  @channel_3[%c0_4, %c0_4] (%results_10[%c0_4, %c0_4, %arg8, %results_14] [%c16, %c8, %c16, %c8] [%c8, %c4096, %c256_2, %c1_3]) {id = 7 : i32} : (memref<2048x256xi8, 1>)
    scf.yield %21 : !air.async.token
  }
  return
}

// CHECK: func.func @func3
// CHECK:  %[[TOK0:.*]], %[[RES0:.*]] = air.execute
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: air.execute_terminator
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2048{{.*}} step %c128{{.*}} iter_args(%[[TOK1:.*]] = %[[TOK0]])
// CHECK-NEXT: air.channel.get async [%[[TOK1]]]  @channel_1
// CHECK:  %[[TOK2:.*]], %[[RES1:.*]] = air.execute
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: air.execute_terminator
// CHECK: %[[TOK6:.*]] = scf.for %{{.*}} = %c0{{.*}} to %c2048{{.*}} step %c128{{.*}} iter_args(%[[TOK3:.*]] = %[[TOK2]])
// CHECK-NEXT: air.channel.get async [%[[TOK3]]]  @channel_2
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2048{{.*}} step %c128{{.*}} iter_args(%[[TOK5:.*]] = %[[TOK0]])
// CHECK-NEXT: air.channel.put async [%[[TOK5]]]  @channel_0
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2048{{.*}} step %c128{{.*}} iter_args(%[[TOK7:.*]] = %[[TOK6]])
// CHECK-NEXT: air.channel.put async [%[[TOK7]]]  @channel_3
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c2048{{.*}} step %c128{{.*}} iter_args(%[[TOK8:.*]] = %[[TOK6]])
// CHECK-NEXT: air.channel.put async [%[[TOK8]]]  @channel_3

func.func @func3(%arg0: memref<2048xi8>, %arg1: memref<2048x1024xi8>, %arg2: memref<1024xi32>) {
  %c4096 = arith.constant 4096 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c256_2 = arith.constant 256 : index
  %c1_3 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_4 = arith.constant 0 : index
  %c2048_5 = arith.constant 2048 : index
  %c128_6 = arith.constant 128 : index
  %async_token_7, %results_8 = air.execute -> (memref<2048xi8, 1>) {
    %alloc = memref.alloc() : memref<2048xi8, 1>
    air.execute_terminator %alloc : memref<2048xi8, 1>
  }
  %6 = scf.for %arg8 = %c0_4 to %c2048_5 step %c128_6 iter_args(%arg9 = %async_token_7) -> (!air.async.token) {
    %21 = air.channel.get async [%arg9]  @channel_1[] (%results_8[%arg8] [%c128_6] [%c1_3]) {id = 4 : i32} : (memref<2048xi8, 1>)
    scf.yield %21 : !air.async.token
  }
  %async_token_9, %results_10 = air.execute -> (memref<2048x256xi8, 1>) {
    %alloc = memref.alloc() : memref<2048x256xi8, 1>
    air.execute_terminator %alloc : memref<2048x256xi8, 1>
  }
  %7 = scf.for %arg8 = %c0_4 to %c2048_5 step %c128_6 iter_args(%arg9 = %async_token_9) -> (!air.async.token) {
    %21 = air.channel.get async [%arg9]  @channel_2[] (%results_10[%arg8, %c0_4] [%c128_6, %c256_2] [%c256_2, %c1_3]) {id = 5 : i32} : (memref<2048x256xi8, 1>)
    scf.yield %21 : !air.async.token
  }
  %async_token_11, %results_12 = air.execute -> (memref<256xi32, 1>) {
    %alloc = memref.alloc() : memref<256xi32, 1>
    air.execute_terminator %alloc : memref<256xi32, 1>
  }
  %8 = scf.for %arg8 = %c0_4 to %c2048_5 step %c128_6 iter_args(%arg9 = %async_token_7) -> (!air.async.token) {
    %21 = air.channel.put async [%arg9]  @channel_0[] (%results_8[%c0_4, %arg8] [%c8, %c16] [%c16, %c1_3]) {id = 6 : i32} : (memref<2048xi8, 1>)
    scf.yield %21 : !air.async.token
  }
  %9 = air.wait_all async [%6, %7, %async_token_11] 
  %async_token_13, %results_14 = air.execute -> (index) {
    air.execute_terminator %c0_4 : index
  }
  %10 = air.wait_all async [%9, %async_token_13] 
  %11 = scf.for %arg8 = %c0_4 to %c2048_5 step %c128_6 iter_args(%arg9 = %10) -> (!air.async.token) {
    %21 = air.channel.put async [%arg9]  @channel_3[%c0_4, %c0_4] (%results_10[%c0_4, %c0_4, %arg8, %results_14] [%c16, %c8, %c16, %c8] [%c8, %c4096, %c256_2, %c1_3]) {id = 7 : i32} : (memref<2048x256xi8, 1>)
    scf.yield %21 : !air.async.token
  }
  %async_token_15, %results_16 = air.execute -> (index) {
    air.execute_terminator %c128_6 : index
  }
  %12 = air.wait_all async [%9, %async_token_15] 
  %13 = scf.for %arg8 = %c0_4 to %c2048_5 step %c128_6 iter_args(%arg9 = %12) -> (!air.async.token) {
    %21 = air.channel.put async [%arg9]  @channel_3[%c1_3, %c0_4] (%results_10[%c0_4, %c0_4, %arg8, %results_16] [%c16, %c8, %c16, %c8] [%c8, %c4096, %c256_2, %c1_3]) {id = 7 : i32} : (memref<2048x256xi8, 1>)
    scf.yield %21 : !air.async.token
  }
  return
}

// Check that RAW, WAR, and WAW dependencies get preserved, while RAR gets dropped.

// CHECK-LABEL: func.func @func4
// CHECK:  %[[GET0:.*]] = air.channel.get async [%{{.*}}]  @channel_3[] (%{{.*}}[] [] [])
// CHECK:  %[[FOR0:.*]] = scf.for %[[FOR0IV:.*]] = %c0{{.*}} to %c8{{.*}} step %c4{{.*}} iter_args(%[[FOR0IARG:.*]] = %[[GET0]])
// CHECK:  %[[FOR1:.*]] = scf.for %[[FOR1IV:.*]] = %c0{{.*}} to %c8{{.*}} step %c4{{.*}} iter_args(%[[FOR1IARG:.*]] = %[[FOR0IARG]])
// CHECK:  %[[PUT0:.*]] = air.channel.put async [%[[FOR1IARG]]]  @channel_0[] (%{{.*}}[%[[FOR0IV]], 
// CHECK:  %[[FOR2:.*]] = scf.for %[[FOR2IV:.*]] = %c1{{.*}} to %c15{{.*}} step %c1{{.*}} iter_args(%[[FOR2IARG:.*]] = %[[FOR1IARG]])
// CHECK:  %[[PUT1:.*]] = air.channel.put async [%[[FOR2IARG]]]  @channel_1[] (%{{.*}}[%[[FOR0IV]], %[[FOR2IV]],
// CHECK:  scf.yield %[[PUT1]] : !air.async.token
// CHECK:  }
// CHECK:  %[[PUT2:.*]] = air.channel.put async [%[[FOR1IARG]]]  @channel_2[] (%{{.*}}[%[[FOR0IV]],
// CHECK:  scf.yield %[[PUT2]] : !air.async.token
// CHECK:  }
// CHECK:  scf.yield %[[FOR1]] : !air.async.token
// CHECK:  }
// CHECK:  %[[GET1:.*]] = air.channel.get async [%[[GET0]], %[[FOR0]]]  @channel_3[] (%{{.*}}[] [] [])

func.func @func4(%arg0: memref<512x512xbf16>, %arg1: memref<512x4096xbf16>, %arg2: memref<512x4096xf32>) {
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c15 = arith.constant 15 : index
  %c4 = arith.constant 4 : index
  %c1_11 = arith.constant 1 : index
  %c16384_12 = arith.constant 16384 : index
  %c32_13 = arith.constant 32 : index
  %c8_14 = arith.constant 8 : index
  %c0_15 = arith.constant 0 : index
  %async_token, %alloc = air.execute -> (memref<8x16x32x32xbf16, 1 : i32>) {
    %alloc = memref.alloc() : memref<8x16x32x32xbf16, 1 : i32>
    air.execute_terminator %alloc : memref<8x16x32x32xbf16, 1 : i32>
  }
  %5 = air.channel.get async [%async_token]  @channel_3[] (%alloc[] [] []) : (memref<8x16x32x32xbf16, 1 : i32>) // WAW
  %7 = air.wait_all async [%async_token, %5] 
  %8 = scf.for %arg10 = %c0_15 to %c8_14 step %c4 iter_args(%arg11 = %7) -> (!air.async.token) {
    %10 = scf.for %arg12 = %c0_15 to %c8_14 step %c4 iter_args(%arg13 = %arg11) -> (!air.async.token) {
      %11 = air.channel.put async [%arg13]  @channel_0[] (%alloc[%arg10, %c0_15, %c0_15, %c0_15, %c0_15, %c0_15] [%c1_11, %c1_11, %c4, %c8_14, %c4, %c8_14] [%c16384_12, %c1024, %c8_14, %c128, %c32_13, %c1_11]) : (memref<8x16x32x32xbf16, 1 : i32>) // RAW
      %12 = air.wait_all async [%arg13, %11]
      %20 = scf.for %arg14 = %c1_11 to %c15 step %c1_11 iter_args(%arg15 = %12) -> (!air.async.token) {
        %31 = air.channel.put async [%arg15]  @channel_1[] (%alloc[%arg10, %arg14, %c0_15, %c0_15, %c0_15, %c0_15] [%c1_11, %c1_11, %c4, %c8_14, %c4, %c8_14] [%c16384_12, %c1024, %c8_14, %c128, %c32_13, %c1_11]) : (memref<8x16x32x32xbf16, 1 : i32>) // RAW, RAR
        scf.yield %31 : !air.async.token
      }
      %21 = air.channel.put async [%arg13, %20]  @channel_2[] (%alloc[%arg10, %c15, %c0_15, %c0_15, %c0_15, %c0_15] [%c1_11, %c1_11, %c4, %c8_14, %c4, %c8_14] [%c16384_12, %c1024, %c8_14, %c128, %c32_13, %c1_11]) : (memref<8x16x32x32xbf16, 1 : i32>) // RAW, RAR
      scf.yield %21 : !air.async.token
    }
    scf.yield %10 : !air.async.token
  }
  %9 = air.channel.get async [%5, %8]  @channel_3[] (%alloc[] [] []) : (memref<8x16x32x32xbf16, 1 : i32>) // WAW, WAR
  return
}

// Check that affinity edges, arising from shared use of symbol refs, are preserved in ssa tokens.

// Here, although the two channel gets do not have data dependency, they have affinity over shared usage of @channel_0.
// Therefore, the dependency between them is preserved in canonicalization.

// CHECK-LABEL: func.func @func5
// CHECK:  %[[GET0:.*]] = air.channel.get async [%{{.*}}]  @channel_0
// CHECK:  %[[GET1:.*]] = air.channel.get async [{{.*}}%[[GET0]]{{.*}}]  @channel_0

func.func @func5() {
  %async_token, %results = air.execute -> (memref<64x64xbf16, 2 : i32>) {
    %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
    air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
  }
  %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
    %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
    air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
  }
  %0 = air.channel.get async [%async_token]  @channel_0[] (%results[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2 : i32>)
  %1 = air.channel.get async [%async_token_0, %0]  @channel_0[] (%results_1[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2 : i32>)
  %async_token_2 = air.execute {
    memref.dealloc %results : memref<64x64xbf16, 2 : i32>
  }
  %async_token_3 = air.execute {
    memref.dealloc %results_1 : memref<64x64xbf16, 2 : i32>
  }
  return
}

// Same as func5, except that the source op is guarded in an if region.
// CHECK-LABEL: func.func @func6
// CHECK:  %[[AIF0:.*]] = affine.if
// CHECK-NEXT:  air.channel.get async [%{{.*}}]  @channel_0
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  else
// CHECK-NEXT:  air.channel.get async [%{{.*}}]  @channel_0
// CHECK-NEXT:  affine.yield
// CHECK:  %[[GET0:.*]] = air.channel.get async [{{.*}}%[[AIF0]]{{.*}}]  @channel_0
func.func @func6() {
  %c2 = arith.constant 2 : index
  %0 = air.herd @herd_0 async  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
    %async_token, %results = air.execute -> (memref<64x64xbf16, 2 : i32>) {
      %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
    }
    %async_token_0, %results_1 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
      %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
    }
    %1 = affine.if affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>()[%arg3, %arg4] -> !air.async.token {
      %3 = air.channel.get async [%async_token]  @channel_0[%arg3, %arg4] (%results[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2 : i32>)
      affine.yield %3 : !air.async.token
    } else {
      %3 = air.channel.get async [%async_token]  @channel_0[%arg3, %arg4] (%results[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2 : i32>)
      affine.yield %3 : !air.async.token
    }
    %2 = air.channel.get async [%async_token_0, %1]  @channel_0[%arg3, %arg4] (%results_1[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2 : i32>)
    %async_token_2 = air.execute {
      memref.dealloc %results : memref<64x64xbf16, 2 : i32>
    }
    %async_token_3 = air.execute {
      memref.dealloc %results_1 : memref<64x64xbf16, 2 : i32>
    }
  }
  return
}
