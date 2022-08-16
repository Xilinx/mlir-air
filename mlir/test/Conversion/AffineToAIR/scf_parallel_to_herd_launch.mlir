// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt -air-par-to-herd %s | FileCheck %s
// CHECK-LABEL: func.func @f0
// CHECK: %[[C0:.*]] = arith.constant 2 : index
// CHECK air.launch_herd tile ({{.*}}, {{.*}}) in ({{.*}}=[[C0]], {{.*}}=[[C0]])
func.func @f0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0,%c0) to (%c2, %c2) step (%c1,%c1) {
    %2 = arith.addi %x, %y : index
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @f1
// CHECK: %[[C1:.*]] = arith.constant 4 : index
// CHECK: %[[C2:.*]] = arith.constant 1 : index
// CHECK air.launch_herd tile ({{.*}}, {{.*}}) in ({{.*}}=[[C1]], {{.*}}=[[C2]])
func.func @f1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %2 = arith.muli %x, %x : index
    scf.yield
  }
  return
}
