// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" | FileCheck %s
//
// Both alternatives are host producers. The device-side clone of their
// index_switch has a dead async result after channel lowering. Remove that
// result before wait legalization while preserving the live host selection.
// This case uses no mixed-memory producer allocation.
// CHECK-LABEL: aie.device(npu2) @seg {
// CHECK-NOT: !air.async.token
// CHECK: func.func @conditional_host
// CHECK: scf.index_switch
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"
// CHECK: scf.yield {{.*}} : !air.async.token
// CHECK: air.channel.put async {{.*}}@x[] {{.*}}metadataArray = [{base = "air_x"
// CHECK: scf.yield {{.*}} : !air.async.token

air.channel @x [1, 1] {channel_type = "npu_dma_packet"}
func.func @conditional_host(%host: memref<8xbf16>, %select: index) {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) args(%h=%host, %s=%select) : memref<8xbf16>, index {
    %choice = scf.index_switch %s -> !air.async.token
    case 0 {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 100 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    }
    default {
      %put = air.channel.put async @x[] (%h[] [] []) {id = 101 : i32} : (memref<8xbf16>)
      scf.yield %put : !air.async.token
    }
    air.wait_all [%choice]
    air.segment @seg {
      %one = arith.constant 1 : index
      air.herd @receiver tile (%tx, %ty) in (%sx=%one, %sy=%one)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %token, %local = air.execute -> (memref<8xbf16, 2>) {
          %buffer = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %buffer : memref<8xbf16, 2>
        }
        air.channel.get @x[] (%local[] [] []) : (memref<8xbf16, 2>)
        %done = air.execute { memref.dealloc %local : memref<8xbf16, 2> }
      }
    }
  }
  return
}
