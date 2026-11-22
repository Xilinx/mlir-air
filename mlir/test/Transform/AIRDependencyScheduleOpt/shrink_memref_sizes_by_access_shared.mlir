//===- shrink_memref_sizes_by_access_shared.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access | FileCheck %s

// A 3x1 herd shares a 2x4 L1 buffer: tiles 0 and 1 each write their row and
// signal tile 2, which sends both rows. The buffer is not per core, so it
// keeps its size; sized to one core's access (1x4), both rows would land on
// row 0.

// CHECK-LABEL: func.func @gather
// CHECK: memref.alloc() : memref<2x4xi32, 2 : i32>
// CHECK-NOT: air.shrinkage
module {
  air.channel @In [3, 1]
  air.channel @Sig [2, 1]
  air.channel @Out [2, 1]
  func.func @gather() {
    %0 = air.segment @s async {
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      %t, %a = air.execute -> (memref<2x4xi32, 2 : i32>) {
        %al = memref.alloc() : memref<2x4xi32, 2 : i32>
        air.execute_terminator %al : memref<2x4xi32, 2 : i32>
      }
      %h = air.herd @h async [%t] tile (%x, %y) in (%sx=%c3, %sy=%c1) args(%buf=%a) : memref<2x4xi32, 2 : i32> {
        %k0 = arith.constant 0 : index
        %k1 = arith.constant 1 : index
        %k2 = arith.constant 2 : index
        %k4 = arith.constant 4 : index
        %s = memref.alloc() : memref<1xi32, 2 : i32>
        %isw = arith.cmpi ult, %x, %k2 : index
        scf.if %isw {
          air.channel.get @In[%x, %y] (%buf[%x, %k0] [%k1, %k4] [%k4, %k1]) : (memref<2x4xi32, 2 : i32>)
          air.channel.put @Sig[%x, %y] (%buf[%x, %k0] [%k1, %k1] [%k4, %k1]) : (memref<2x4xi32, 2 : i32>)
        } else {
          air.channel.get @Sig[%k0, %y] (%s[] [] []) : (memref<1xi32, 2 : i32>)
          air.channel.get @Sig[%k1, %y] (%s[] [] []) : (memref<1xi32, 2 : i32>)
          %r0 = arith.subi %x, %k2 : index
          %r1 = arith.subi %x, %k1 : index
          air.channel.put @Out[%k0, %y] (%buf[%r0, %k0] [%k1, %k4] [%k4, %k1]) : (memref<2x4xi32, 2 : i32>)
          air.channel.put @Out[%k1, %y] (%buf[%r1, %k0] [%k1, %k4] [%k4, %k1]) : (memref<2x4xi32, 2 : i32>)
        }
        memref.dealloc %s : memref<1xi32, 2 : i32>
      }
    }
    return
  }
}
