// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt -air-par-to-herd %s | FileCheck %s
// CHECK-LABEL: func.func @foo
// CHECK: %[[C0:.*]] = arith.constant 1 : index
// CHECK air.herd tile ({{.*}}, {{.*}}) in ({{.*}}=[[C0]], {{.*}}=[[C0]])
module  {
  func.func @foo()  {
    affine.parallel (%x,%y) = (0,0) to (1,1) {
      %2 = arith.addi %x, %y : index
      affine.yield
    }
    return
  }
}