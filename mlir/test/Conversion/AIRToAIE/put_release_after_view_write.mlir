//===- put_release_after_view_write.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2" | FileCheck %s

// A view of %a, made and used before the send, is written after it: the send's release
// waits for that write.

// CHECK-LABEL: aie.core
// CHECK: func.call @use(
// CHECK-NOT: Release
// CHECK: func.call @use_half
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release

air.channel @to_core [1, 1]
air.channel @from_core [1, 1]
func.func private @use(memref<64xbf16, 2>, memref<64xbf16, 2>) attributes {link_with = "use.o"}
func.func private @use_half(memref<32xbf16, strided<[1]>, 2>, memref<64xbf16, 2>) attributes {link_with = "use.o"}
func.func @view_written_after_send(%x: memref<64xbf16>, %y: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%a0, %a1) in (%s0=%c1, %s1=%c1) args(%xx=%x, %yy=%y) : memref<64xbf16>, memref<64xbf16> {
    air.channel.put @to_core[] (%xx[] [] []) : (memref<64xbf16>)
    air.channel.get @from_core[] (%yy[] [] []) : (memref<64xbf16>)
    air.segment @seg {
      %one = arith.constant 1 : index
      air.herd @h tile(%tx, %ty) in (%sx=%one, %sy=%one) attributes {link_with = "use.o", x_loc = 0 : i64, y_loc = 2 : i64} {
        %a = memref.alloc() : memref<64xbf16, 2>
        %b = memref.alloc() : memref<64xbf16, 2>
        %v = memref.subview %a[0] [32] [1] : memref<64xbf16, 2> to memref<32xbf16, strided<[1]>, 2>
        air.channel.get @to_core[%tx, %ty] (%b[] [] []) : (memref<64xbf16, 2>)
        func.call @use_half(%v, %b) : (memref<32xbf16, strided<[1]>, 2>, memref<64xbf16, 2>) -> ()
        func.call @use(%a, %b) : (memref<64xbf16, 2>, memref<64xbf16, 2>) -> ()
        air.channel.put @from_core[%tx, %ty] (%a[] [] []) : (memref<64xbf16, 2>)
        func.call @use_half(%v, %b) : (memref<32xbf16, strided<[1]>, 2>, memref<64xbf16, 2>) -> ()
        memref.dealloc %a : memref<64xbf16, 2>
        memref.dealloc %b : memref<64xbf16, 2>
      }
    }
  }
  return
}
