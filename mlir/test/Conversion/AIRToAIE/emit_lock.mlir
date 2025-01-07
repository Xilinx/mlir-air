//===- emit_lock.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='emit-herd-lock=true' -split-input-file | FileCheck %s
// RUN: air-opt %s -air-to-aie='emit-herd-lock=false device=npu1_4col row-offset=2' -split-input-file | FileCheck %s --check-prefix=NPU1

// CHECK-LABEL: aie.device(xcvc1902)
// CHECK:  %[[VAL_0:.*]] = aie.tile
// CHECK:  %[[VAL_2:.*]] = aie.lock(%[[VAL_0]],
// CHECK:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[VAL_2]], Acquire, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    aie.use_lock(%[[VAL_2]], Release, 0)
// CHECK:    aie.end

// NPU1-LABEL: aie.device(npu1_4col)
// NPU1:  %[[VAL_0:.*]] = aie.tile
// NPU1:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// NPU1:    cf.br ^bb1
// NPU1:  ^bb1:
// NPU1:    cf.br ^bb2
// NPU1:  ^bb2:
// NPU1:    aie.end

module {
  func.func @func1() -> () {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) {
    }
    return
  }
}

// -----

// CHECK-LABEL: aie.device(xcvc1902)
// CHECK:  %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// CHECK:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// CHECK:  %[[HERD_LOCK:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Acquire, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Release, 0)
// CHECK:    aie.end

// NPU1-LABEL: aie.device(npu1_4col)
// NPU1:  %[[VAL_0:.*]] = aie.tile(1, 2)
// NPU1:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[LOCK_1:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// NPU1:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// NPU1:    cf.br ^bb1
// NPU1:  ^bb1:
// NPU1:    cf.br ^bb2
// NPU1:  ^bb2:
// NPU1:    aie.use_lock(%[[LOCK_1]], AcquireGreaterEqual, 1)
// NPU1:    aie.use_lock(%[[LOCK_0]], Release, 1)
// NPU1:    aie.end

module {
  air.channel @channel_2 [1, 1]
  func.func @func2(%arg0 : memref<1024xi32>) -> () {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) {
      %alloc = memref.alloc() : memref<1024xi32, 2>
      air.channel.get @channel_2[] (%alloc[] [] []) : (memref<1024xi32, 2>)
      memref.dealloc %alloc : memref<1024xi32, 2>
    }
    return
  }
}

// -----

// CHECK-LABEL: aie.device(xcvc1902)
// CHECK:  %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// CHECK:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// CHECK:  %[[HERD_LOCK:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Acquire, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK-DAG:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK-DAG:    aie.use_lock(%[[HERD_LOCK]], Release, 0)
// CHECK:    aie.end

// NPU1-LABEL: aie.device(npu1_4col)
// NPU1:  %[[VAL_0:.*]] = aie.tile(1, 2)
// NPU1:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[LOCK_1:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// NPU1:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// NPU1:    cf.br ^bb1
// NPU1:  ^bb1:
// NPU1:    cf.br ^bb2
// NPU1:  ^bb2:
// NPU1:    aie.use_lock(%[[LOCK_1]], AcquireGreaterEqual, 1)
// NPU1:    aie.use_lock(%[[LOCK_0]], Release, 1)
// NPU1:    aie.end

module {
  air.channel @channel_2 [1, 1]
  func.func @func3(%arg0 : memref<1024xi32>) -> () {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    %alloc = memref.alloc() : memref<1024xi32, 2>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args (%arg1=%alloc) : memref<1024xi32, 2> {
      air.channel.get @channel_2[] (%arg1[] [] []) : (memref<1024xi32, 2>)
    }
    memref.dealloc %alloc : memref<1024xi32, 2>
    return
  }
}

// -----

// CHECK-LABEL: aie.device(xcvc1902)
// CHECK:  %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// CHECK:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// CHECK:  %[[HERD_LOCK:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Acquire, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK:    scf.for
// CHECK:      aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK:      aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK:    }
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Release, 0)
// CHECK:    aie.end

// NPU1-LABEL: aie.device(npu1_4col)
// NPU1:  %[[VAL_0:.*]] = aie.tile(1, 2)
// NPU1:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[LOCK_1:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// NPU1:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// NPU1:    cf.br ^bb1
// NPU1:  ^bb1:
// NPU1:    cf.br ^bb2
// NPU1:  ^bb2:
// NPU1:    aie.use_lock(%[[LOCK_1]], AcquireGreaterEqual, 1)
// NPU1:    aie.use_lock(%[[LOCK_0]], Release, 1)
// NPU1:    scf.for
// NPU1:      aie.use_lock(%[[LOCK_1]], AcquireGreaterEqual, 1)
// NPU1:      aie.use_lock(%[[LOCK_0]], Release, 1)
// NPU1:    }
// NPU1:    aie.end

module {
  air.channel @channel_2 [1, 1]
  func.func @func4(%arg0 : memref<1024xi32>) -> () {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    scf.for %arg3 = %c0 to %c8 step %c1 {
      air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    }
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c8_0 = arith.constant 8 : index
      %alloc = memref.alloc() : memref<1024xi32, 2>
      air.channel.get @channel_2[] (%alloc[] [] []) : (memref<1024xi32, 2>)
      memref.dealloc %alloc : memref<1024xi32, 2>
      scf.for %arg4 = %c0_0 to %c8_0 step %c1_0 {
        air.channel.get @channel_2[] (%alloc[] [] []) : (memref<1024xi32, 2>)
        memref.dealloc %alloc : memref<1024xi32, 2>
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: aie.device(xcvc1902)
// CHECK:  %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// CHECK:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// CHECK:  %[[HERD_LOCK:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32}
// CHECK:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Acquire, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK:    scf.for
// CHECK:      aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK:      aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK:    }
// CHECK:    aie.use_lock(%[[HERD_LOCK]], Release, 0)
// CHECK:    aie.end

// NPU1-LABEL: aie.device(npu1_4col)
// NPU1:  %[[VAL_0:.*]] = aie.tile(1, 2)
// NPU1:  %[[LOCK_0:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[LOCK_1:.*]] = aie.lock(%[[VAL_0]],
// NPU1:  %[[BUF_0:.*]] = aie.buffer(%[[VAL_0]]) {{.*}} : memref<1024xi32, 2> 
// NPU1:  %[[VAL_3:.*]] = aie.core(%[[VAL_0]]) {
// NPU1:    cf.br ^bb1
// NPU1:  ^bb1:
// NPU1:    cf.br ^bb2
// NPU1:  ^bb2:
// NPU1:    aie.use_lock(%[[LOCK_1]], AcquireGreaterEqual, 1)
// NPU1:    aie.use_lock(%[[LOCK_0]], Release, 1)
// NPU1:    scf.for
// NPU1:      aie.use_lock(%[[LOCK_1]], AcquireGreaterEqual, 1)
// NPU1:      aie.use_lock(%[[LOCK_0]], Release, 1)
// NPU1:    }
// NPU1:    aie.end

module {
  air.channel @channel_2 [1, 1]
  func.func @func5(%arg0 : memref<1024xi32>) -> () {
    %herd_cols = arith.constant 1 : index
    %herd_rows = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    scf.for %arg3 = %c0 to %c8 step %c1 {
      air.channel.put @channel_2[] (%arg0[] [] []) : (memref<1024xi32>)
    }
    %alloc = memref.alloc() : memref<1024xi32, 2>
    air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args (%arg1=%alloc) : memref<1024xi32, 2> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c8_0 = arith.constant 8 : index
      air.channel.get @channel_2[] (%arg1[] [] []) : (memref<1024xi32, 2>)
      scf.for %arg4 = %c0_0 to %c8_0 step %c1_0 {
        air.channel.get @channel_2[] (%arg1[] [] []) : (memref<1024xi32, 2>)
      }
    }
    memref.dealloc %alloc : memref<1024xi32, 2>
    return
  }
}
