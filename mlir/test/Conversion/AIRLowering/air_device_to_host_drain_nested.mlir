//===- air_device_to_host_drain_nested.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std -split-input-file -verify-diagnostics | FileCheck %s

// A read of host memory that depends on a drain in the same case of an
// scf.index_switch (a decode wave's KV append and readback) is preceded by a
// blocking wait on that drain, inside the case.

// CHECK-LABEL: func.func @switch
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc}
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @out [1, 1]
  air.channel @in [1, 1]
  func.func @switch(%buf: memref<256xi32>, %sel: index) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf, %s0=%sel) : memref<256xi32>, index {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %r = scf.index_switch %s0 -> !air.async.token
      case 0 {
        %d = air.channel.get async  @out[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc} : (memref<256xi32>)
        %rb = air.channel.put async [%d]  @in[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAlloc} : (memref<256xi32>)
        %y = air.wait_all async [%d, %rb]
        scf.yield %y : !air.async.token
      }
      default {
        scf.yield %t0 : !air.async.token
      }
      %e = air.wait_all async [%r] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// In a loop body a drain left outstanding by one iteration would take the await
// meant for the next, so the read is not ordered there; it is diagnosed.

// CHECK-LABEL: func.func @loop
// CHECK: scf.for
// CHECK-NOT: airrt.wait_all %{{[0-9]+}}{{$}}
// CHECK: scf.yield

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc1(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc1(%t, MM2S, 0)
  } {sym_name = "seg1"}
  air.channel @out1 [1, 1]
  air.channel @in1 [1, 1]
  func.func @loop(%buf: memref<256xi32>, %sel: index) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf, %s0=%sel) : memref<256xi32>, index {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %c2 = arith.constant 2 : index
      %r = scf.for %it = %c0 to %c2 step %c1_l iter_args(%tk = %t0) -> (!air.async.token) {
        %d = air.channel.get async [%tk]  @out1[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc1} : (memref<256xi32>)
        // expected-warning@+2 {{this read of host memory depends on a device-to-host drain that cannot be awaited before it here}}
        // expected-note@+1 {{it is in a loop body}}
        %rb = air.channel.put async [%d]  @in1[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAlloc1} : (memref<256xi32>)
        %y = air.wait_all async [%d, %rb]
        scf.yield %y : !air.async.token
      }
      %e = air.wait_all async [%r] {air.launch_end}
      %s = air.segment @seg1 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h1 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in1[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out1[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// A read in a case that depends on a drain issued before the scf.index_switch:
// the case cannot tell which of the channel's tasks are still outstanding when
// it runs, so the read is not ordered there; it is diagnosed.

// CHECK-LABEL: func.func @outside
// CHECK: case 0 {
// CHECK-NOT: airrt.wait_all %{{[0-9]+}}{{$}}
// CHECK: scf.yield

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAllocoutside(%t, S2MM, 0)
    aie.shim_dma_allocation @inAllocoutside(%t, MM2S, 0)
  } {sym_name = "segoutside"}
  air.channel @outoutside [1, 1]
  air.channel @inoutside [1, 1]
  func.func @outside(%buf: memref<256xi32>, %sel: index) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf, %s0=%sel) : memref<256xi32>, index {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %d = air.channel.get async  @outoutside[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAllocoutside} : (memref<256xi32>)
      %r = scf.index_switch %s0 -> !air.async.token
      case 0 {
        // expected-warning@+2 {{this read of host memory depends on a device-to-host drain that cannot be awaited before it here}}
        // expected-note@+1 {{the drain is outside its block}}
        %rb = air.channel.put async [%d]  @inoutside[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAllocoutside} : (memref<256xi32>)
        %y = air.wait_all async [%d, %rb]
        scf.yield %y : !air.async.token
      }
      default {
        scf.yield %d : !air.async.token
      }
      %e = air.wait_all async [%r] {air.launch_end}
      %s = air.segment @segoutside async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @houtside async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @inoutside[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @outoutside[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Nesting is followed down: a drain and its read inside an scf.if inside an
// scf.index_switch case get the wait in the scf.if.

// CHECK-LABEL: func.func @deep
// CHECK: case 0 {
// CHECK: scf.if
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAllocdeep}
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAllocdeep}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAllocdeep(%t, S2MM, 0)
    aie.shim_dma_allocation @inAllocdeep(%t, MM2S, 0)
  } {sym_name = "segdeep"}
  air.channel @outdeep [1, 1]
  air.channel @indeep [1, 1]
  func.func @deep(%buf: memref<256xi32>, %sel: index) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf, %s0=%sel) : memref<256xi32>, index {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %t0 = air.wait_all async
      %c0_i = arith.constant 0 : index
      %r = scf.index_switch %s0 -> !air.async.token
      case 0 {
        %cond = arith.cmpi eq, %i, %c0_i : index
        %ri = scf.if %cond -> (!air.async.token) {
          %d = air.channel.get async  @outdeep[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAllocdeep} : (memref<256xi32>)
          %rb = air.channel.put async [%d]  @indeep[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAllocdeep} : (memref<256xi32>)
          %y = air.wait_all async [%d, %rb]
          scf.yield %y : !air.async.token
        } else {
          scf.yield %t0 : !air.async.token
        }
        scf.yield %ri : !air.async.token
      }
      default {
        scf.yield %t0 : !air.async.token
      }
      %e = air.wait_all async [%r] {air.launch_end}
      %s = air.segment @segdeep async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @hdeep async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @indeep[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @outdeep[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
