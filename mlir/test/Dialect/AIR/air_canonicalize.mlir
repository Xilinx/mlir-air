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
